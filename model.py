import torch
import torch.nn as nn
import torch.nn.functional as F
class DRAWAttentionParams(nn.Module):
    def __init__(self, input_size, img_h, img_w):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w

        self.g_x = nn.Linear(input_size, 1)
        self.g_y = nn.Linear(input_size, 1)
        self.log_sigma = nn.Linear(input_size, 1)
        self.log_delta = nn.Linear(input_size, 1)
        self.log_gamma = nn.Linear(input_size, 1)

    def forward(self, h_dec, N):
        g_x = self.g_x(h_dec)
        g_y = self.g_y(h_dec)
        sigma = torch.exp(self.log_sigma(h_dec))
        delta = torch.exp(self.log_delta(h_dec))
        gamma = torch.exp(self.log_gamma(h_dec))
        g_x = (self.img_h + 1.) / 2. * (g_x + 1.)
        g_y = (self.img_w + 1.) / 2. * (g_y + 1.)
        delta = (max(self.img_h, self.img_w) - 1.) / (N - 1.) * delta
        return g_x, g_y, sigma, delta, gamma

# TODO SEPARATE READ AND WRITE ATTENTION
class DRAW(nn.Module):
    def __init__(self, lstm_hidden, z_size, T, img_h, img_w, read_size=None, write_size=None, use_gpu=True):
        # set class vars
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.z_size = z_size
        self.T = T
        self.img_h = img_h # A in the paper
        self.img_w = img_w # B in the paper
        self.input_len = img_h * img_w
        self.read_size = read_size # N in the paper
        self.write_size = write_size # also N in the paper (attention is applied at 2 moments)
        self.device = "cuda" if use_gpu else "cpu"
        self.decoder_loss_fn = nn.BCELoss(reduction='sum')
        self.eps = 1e-7

        self.sigmas = [None] * self.T
        self.logsigmas = [None] * self.T
        self.mus = [None] * self.T
        self.attention_history = [None] * self.T
        self.canvas_history = [None] * self.T

        if (self.write_size is None) == (self.read_size is None):
            self.need_attention = self.write_size is not None
        else:
            raise ValueError("read and write size must either both be None or int")


        # step 0 canvas
        self.c_0 = nn.Parameter(torch.randn(1, self.input_len,
                                requires_grad=True))
        # self.c_0 = nn.Parameter(torch.zeros(1, self.input_len,
        #                         requires_grad=True))

        # attention paramters weight vectors
        if self.need_attention:
            self.read_attention_params_matrices = DRAWAttentionParams(self.lstm_hidden, self.img_h, self.img_w)
            self.write_attention_params_matrices = DRAWAttentionParams(self.lstm_hidden, self.img_h, self.img_w)

        # Encoder
        self.h_0_enc = nn.Parameter(torch.randn(self.lstm_hidden,
                                    requires_grad=True))
        if not self.need_attention:
            encoder_input_size = self.input_len * 2 + self.lstm_hidden
        else:
            encoder_input_size = int(self.read_size)**2 * 2 + self.lstm_hidden
        self.encoder_cell = nn.LSTMCell(encoder_input_size,
                                        self.lstm_hidden)

        # Q
        self.mu_matrix = nn.Linear(self.lstm_hidden, z_size)
        self.sigma_matrix = nn.Linear(self.lstm_hidden, z_size)

        # Decoder
        self.h_0_dec = nn.Parameter(torch.randn(self.lstm_hidden,
                                    requires_grad=True))
        self.decoder_cell = nn.LSTMCell(self.z_size,
                                        self.lstm_hidden)

        if self.need_attention:
            self.writer_matrix = nn.Linear(self.lstm_hidden, int(self.write_size)**2)
        else:
            self.writer_matrix = nn.Linear(self.lstm_hidden, self.input_len)

    def _sample_Q(self, h_enc):
        batch_size = h_enc.size(0)
        z = torch.randn(batch_size, self.z_size, device=self.device)
        mu = self.mu_matrix(h_enc)
        logsigma = self.sigma_matrix(h_enc)
        sigma = torch.exp(logsigma)

        self.mus[self.t] = mu
        self.sigmas[self.t] = sigma
        self.logsigmas[self.t] = logsigma

        return z * sigma + mu

    def _get_filter_mus(self, g, delta, N):
        bsz = g.size(0)
        i_s = torch.arange(N, device=self.device).float().unsqueeze(0).repeat(bsz, 1) # create matrix of ranges bsz * N
        mu = g + (i_s - N / 2. - 0.5) * delta
        return mu

    def _get_filter_bank(self, mu, sigma, a):
        bsz = mu.size(0)
        N = mu.size(1)
        # shape a * N * bsz this is done for broadcasting
        F = torch.arange(a, device=self.device).float().unsqueeze(-1).unsqueeze(-1).repeat(1, N, bsz)
        # transpose mu and sigma for broadcasting
        mu = mu.transpose(0, 1)
        sigma = sigma.transpose(0, 1)
        F = torch.exp(
            -1. * (F - mu)**2 / (2. * sigma)
        )
        F = F / (F.detach().sum(dim=0) + self.eps) # normalize filter
        F = F.permute(2, 1, 0)
        return F

    def get_filter_banks(self, h_dec, N, read=True):
        if read:
            attention_params_matrices = self.read_attention_params_matrices
        else:
            attention_params_matrices = self.write_attention_params_matrices
        g_x, g_y, sigma, delta, gamma = attention_params_matrices(h_dec, N)
        if self.save_attentions and read:
            self.attention_history[self.t] = {
                        "g_x": g_x.cpu().detach().numpy(),
                        "g_y": g_y.cpu().detach().numpy(),
                        "delta": delta.cpu().detach().numpy(),
            }
        mu_x = self._get_filter_mus(g_x, delta, N) # shape bsz * N
        mu_y = self._get_filter_mus(g_y, delta, N) # shape bsz * N

        F_x = self._get_filter_bank(mu_x, sigma, self.img_h)
        F_y = self._get_filter_bank(mu_y, sigma, self.img_w)
        return gamma, F_x, F_y


    def decoder_loss(self, x, y):
        return self.decoder_loss_fn(x, y) / x.size(0) # divide by batch size

    def latent_loss(self, x):
        loss = 0.
        for (mu, sigma, logsigma) in zip(self.mus, self.sigmas, self.logsigmas):
            loss += (mu**2 + sigma**2 - 2* logsigma) / 2
        loss -= self.T / 2
        loss = torch.sum(loss) / x.size(0)
        return loss


    def loss(self, x, y):
        return self.decoder_loss(x, y), self.latent_loss(x)

    def forward(self, x, save_attentions=False):
        self.save_attentions = save_attentions
        batch_size = x.size(0)
        c_t = self.c_0

        # learnable LSTM params
        h_t_dec = self.h_0_dec.repeat(batch_size, 1)
        h_t_enc = self.h_0_enc.repeat(batch_size, 1)

        # non-learnable LSTM params (initial cell states)
        c_dec = torch.zeros(batch_size, self.lstm_hidden, device=self.device)
        c_enc = torch.zeros(batch_size, self.lstm_hidden, device=self.device)
        for self.t in range(self.T):
            x_hat = x - torch.sigmoid(c_t)
            r_t = self.read(x, x_hat, h_t_dec)
            h_t_enc, c_enc = self.encoder_cell(
                torch.cat((r_t, h_t_dec), dim=-1),
                (h_t_enc, c_enc)
            )
            z_t = self._sample_Q(h_t_enc)
            h_t_dec, c_dec = self.decoder_cell(
                z_t,
                (h_t_dec, c_dec)
            )
            c_t = c_t + self.write(h_t_dec)
            if self.save_attentions:
                self.canvas_history[self.t] = torch.sigmoid(c_t.cpu().detach()).numpy()
        if self.save_attentions:
            self.save_attentions = False
            return self.canvas_history, self.attention_history
        self.save_attentions = False
        return torch.sigmoid(c_t)


    def generate(self, batch_size, save_history=False):
        c_dec = torch.zeros(batch_size, self.lstm_hidden, device=self.device)
        h_dec = self.h_0_dec.data.repeat(batch_size, 1).detach()
        canvas = self.c_0.data.repeat(batch_size, 1).detach()
        history = [None] * self.T
        for t in range(self.T):
            z = torch.randn(batch_size, self.z_size, device=self.device)
            h_dec, c_dec = self.decoder_cell(z, (h_dec, c_dec))
            canvas += self.write(h_dec)
            if save_history:
                history[t] = torch.sigmoid(canvas).cpu().detach()
        if save_history:
            return history
        return torch.sigmoid(canvas)

    def read(self, x, x_hat, h_dec):
        if self.need_attention:
            bsz = x.size(0)
            gamma, F_x, F_y = self.get_filter_banks(h_dec, self.read_size)
            x = x.view(bsz, self.img_h, self.img_w)
            x_hat = x.view(bsz, self.img_h, self.img_w)
            def apply_F(x, F_y, F_x, gamma):
                x = torch.bmm(torch.bmm(F_y, x), F_x.transpose(1, 2))
                x = x.permute(2, 1, 0)
                gamma = gamma.transpose(0, 1)
                x = x * gamma
                x = x.permute(2, 1, 0)
                return x
            x = apply_F(x, F_y, F_x, gamma)
            x_hat = apply_F(x_hat, F_y, F_x, gamma)
            x = x.view(bsz, self.read_size**2)
            x_hat = x_hat.view(bsz, self.read_size**2)
        return torch.cat((x, x_hat), dim=-1)

    def write(self, h_dec):
        w = self.writer_matrix(h_dec)
        if self.need_attention:
            bsz = h_dec.size(0)
            w = w.view(bsz, self.write_size, self.write_size)
            gamma, F_x, F_y = self.get_filter_banks(h_dec, self.write_size, read=False)
            w = torch.bmm(torch.bmm(F_y.transpose(1, 2), w), F_x)
            w = w.permute(2, 1, 0)
            gamma = 1. / gamma.transpose(0, 1)
            w = w * gamma
            w = w.permute(2, 1, 0)
            w = w.view(bsz, self.img_h * self.img_w)
        return w

