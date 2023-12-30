import torch
import torch.nn as nn
import torchaudio
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.c0=nn.Conv2d(3, 25, kernel_size=3, stride=1, padding=1)
        self.c1=nn.Conv2d(25, 80, kernel_size=3, stride=1, padding=1)
        self.enc=torchaudio.models.Conformer()
    def encode(self, frameseq):
        latent=self.c0(frameseq)
        latent=self.enc(latent)
        return latent

class Decoder_L(nn.Module):
    def __init__(self):
        self.dlay=nn.TransformerDecoderLayer()
        self.dec=nn.TransformerDecoder()
class Decoder_A(nn.Module):
    def __init__(self):
        self.dlay=nn.TransformerDecoderLayer()
        self.dec=nn.TransformerDecoder()


class Reconstruct(nn.Module):
    def __init__(self):
        self.dec=nn.TransformerDecoder()
        self.upconv0=nn.ConvTranspose2d()
        self.upconv1=nn.ConvTranspose2d()


def train_Language_Embedding(textembedder, data):
    enc=Encoder()
    enc_l=Decoder_L()
    textemb=textembedder
    embedding=enc(data)
    embedding=enc_l(embedding)
    sub=textemb(embedding)
    optim=torch.optim.Adam(list(enc.parameters())+list(enc_l.parameters()), 0.0001)
    lossfn=nn.MSELoss()
    loss=lossfn(sub, embedding)
    optim.zero_grad()
    loss.backward()
    optim.step()



def train_reconstruction():
    enc=Encoder()
    enc_a=Decoder_A()
    enc_l=Decoder_L()
    rec=Reconstruct()
    data=enc(data)
    acoustic_feature=enc_a(data)
    emb=enc_l(data)
    out=rec(acoustic_feature, emb)

    lossfn=nn.MSELoss()
    optim=torch.optim.Adam([], 0.0001)
    loss=lossfn(out, data)
    optim.zero_grad()
    loss.backward()
    optim.step()

def train_backtranslate():