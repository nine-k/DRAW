import torch
import torch.nn as nn
import torch.nn.functional as F

class DRAW(nn.Module):
    def __init__(self, lstm_hidden, z_size, T, img_h, img_w, read_size=None, write_size=None, use_gpu=HAS_CUDA):
        # set class vars
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.z_size = z_size
        self.T = T
        self.img_h = img_h
        self.img_w = img_w
        self.input_len = img_h * img_w
        self.read_size = read_size
        self.write_size = write_size
        self.device = "cuda" if use_gpu else "cpu"
        self.decoder_loss_fn = nn.BCELoss(reduction='sum')
        
        self.sigmas = [None] * self.T
        self.logsigmas = [None] * self.T
        self.mus = [None] * self.T

        if (self.write_size is None) == (self.read_size is None):
            self.need_attention = self.write_size is not None
        else:
            raise ValueError("read and write size must either both be None or int")


        # set model
        self.c_0 = nn.Parameter(torch.randn(1, self.input_len,
                                requires_grad=True))
        
        if self.need_attention:
            pass #TODO add reader

        # Encoder
        self.h_0_enc = nn.Parameter(torch.randn(self.lstm_hidden,
                                    requires_grad=True))
        self.encoder_cell = nn.LSTMCell(self.input_len * 2 + self.lstm_hidden,
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
            pass #TODO add writer
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

    def step(self):
        pass

    def decoder_loss(self, x, y):
        return self.decoder_loss_fn(x, y) / x.size(0) # divide by batch size

    def latent_loss(self):
        loss = 0.
        for (mu, sigma, logsigma) in zip(self.mus, self.sigmas, self.logsigmas):
            loss += (mu**2 + sigma**2 - 2* logsigma) / 2
        loss -= self.T / 2
        loss = torch.sum(loss) / x.size(0)
        return loss


    def loss(self, x, y):
        return self.decoder_loss(x, y), self.latent_loss()

    def forward(self, x):
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
            raise NotImplementedError("not implemented yet")
        else:
            return torch.cat((x, x_hat), dim=-1)
    
    def write(self, h_dec):
        if self.need_attention:
            raise NotImplementedError("not implemented yet")
        else:
            return self.writer_matrix(h_dec)
