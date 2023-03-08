import torch
import torch.nn as nn

class LinBnDrop(nn.Sequential):
    """Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"""
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

        
class Encoder(nn.Module):
    """Encoder for CITE-seq data"""
    def __init__(self, nfeatures_modality1=10703, nfeatures_modality2=192, hidden_modality1=185,  hidden_modality2=15, z_dim=128):
        super().__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.encoder_modality1 = LinBnDrop(nfeatures_modality1, hidden_modality1, p=0.2, act=nn.ReLU())
        self.encoder_modality2 = LinBnDrop(nfeatures_modality2, hidden_modality2, p=0.2, act=nn.ReLU())
        self.encoder = LinBnDrop(hidden_modality1 + hidden_modality2, z_dim,  p=0.2, act=nn.ReLU())
        self.weights_modality1 = nn.Parameter(torch.rand((1,nfeatures_modality1)) * 0.001, requires_grad=True)
        self.weights_modality2 = nn.Parameter(torch.rand((1,nfeatures_modality2)) * 0.001, requires_grad=True)

    def forward(self, x):
        x_modality1 = self.encoder_modality1(x[:, :self.nfeatures_modality1]*self.weights_modality1)
        x_modality2 = self.encoder_modality2(x[:, self.nfeatures_modality1:]*self.weights_modality2)
        x = torch.cat([x_modality1, x_modality2], 1)
        x = self.encoder(x)
        return x
    
    
class Encoder_3modality(nn.Module):
    """Encoder for CITE-seq data"""
    def __init__(self, nfeatures_modality1=10703, nfeatures_modality2=192, nfeatures_modality3=10703, hidden_modality1=185,  hidden_modality2=15, hidden_modality3=185, z_dim=128):
        super().__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.nfeatures_modality3 = nfeatures_modality3
        self.encoder_modality1 = LinBnDrop(nfeatures_modality1, hidden_modality1, p=0.2, act=nn.ReLU())
        self.encoder_modality2 = LinBnDrop(nfeatures_modality2, hidden_modality2, p=0.2, act=nn.ReLU())
        self.encoder_modality3 = LinBnDrop(nfeatures_modality3, hidden_modality3, p=0.2, act=nn.ReLU())
        self.encoder = LinBnDrop(hidden_modality1 + hidden_modality2+ hidden_modality3, z_dim,  p=0.2, act=nn.ReLU())
        self.weights_modality1 = nn.Parameter(torch.rand((1,nfeatures_modality1)) * 0.001, requires_grad=True)
        self.weights_modality2 = nn.Parameter(torch.rand((1,nfeatures_modality2)) * 0.001, requires_grad=True)
        self.weights_modality3 = nn.Parameter(torch.rand((1,nfeatures_modality3)) * 0.001, requires_grad=True)

    def forward(self, x):
        x_modality1 = self.encoder_modality1(x[:, :self.nfeatures_modality1]*self.weights_modality1)
        x_modality2 = self.encoder_modality2(x[:, self.nfeatures_modality1:(self.nfeatures_modality1+self.nfeatures_modality2)]*self.weights_modality2)
        x_modality3 = self.encoder_modality3(x[:, (self.nfeatures_modality1+self.nfeatures_modality2):]*self.weights_modality3)
        x = torch.cat([x_modality1, x_modality2, x_modality3], 1)
        x = self.encoder(x)
        return x
    

class Decoder(nn.Module):
    """Decoder for for 2 modalities data (citeseq data and shareseq data) """
    def __init__(self, nfeatures_modality1=10703, nfeatures_modality2=192,  hidden_modality1=185,  hidden_modality2=15, z_dim=128):
        super().__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.decoder11 = nn.Sequential(LinBnDrop(z_dim, 100, act=nn.ReLU()),
                                      )
        self.decoder12 = nn.Sequential(LinBnDrop(100, nfeatures_modality1, act=nn.ReLU()))
        self.decoder21 = nn.Sequential(LinBnDrop(z_dim, 100,  act=nn.ReLU()),
                                      )
        self.decoder22 = nn.Sequential(LinBnDrop(100, nfeatures_modality2,  act=nn.ReLU()))

    def forward(self, x):
        x_rna = self.decoder12(self.decoder11(x))
        x_adt = self.decoder22(self.decoder21(x))
        x = torch.cat((x_rna,x_adt),1)
        return x
    
class Decoder_3modality(nn.Module):
    """Decoder for for 2 modalities data (citeseq data and shareseq data) """
    def __init__(self, nfeatures_modality1=10703, nfeatures_modality2=192, nfeatures_modality3=192,  hidden_modality1=185,  hidden_modality2=15,  hidden_modality3=15, z_dim=128):
        super().__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.nfeatures_modality3 = nfeatures_modality3
        self.decoder11 = nn.Sequential(LinBnDrop(z_dim, 100, act=nn.ReLU()),
                                      )
        self.decoder12 = nn.Sequential(LinBnDrop(100, nfeatures_modality1, act=nn.ReLU()))
        self.decoder21 = nn.Sequential(LinBnDrop(z_dim, 100,  act=nn.ReLU()),
                                      )
        self.decoder22 = nn.Sequential(LinBnDrop(100, nfeatures_modality2,  act=nn.ReLU()))
        self.decoder31 = nn.Sequential(LinBnDrop(z_dim, 100,  act=nn.ReLU()),
                                      )
        self.decoder32 = nn.Sequential(LinBnDrop(100, nfeatures_modality3,  act=nn.ReLU()))

    def forward(self, x):
        x_rna = self.decoder12(self.decoder11(x))
        x_adt = self.decoder22(self.decoder21(x))
        x_atac = self.decoder32(self.decoder31(x))
        x = torch.cat((x_rna,x_adt,x_atac),1)
        return x

    
class Classifier(nn.Module):
    def __init__(self, nfeatures_modality1=10703, nfeatures_modality2=192,  hidden_modality1=185,  hidden_modality2=15, z_dim=128,classify_dim=17):
        super().__init__()
        self.classify = nn.Linear(z_dim, classify_dim)

    def forward(self, x):
        x_cty = self.classify(x)
        return x_cty

    
class DeepMerge(nn.Module):
    def __init__(self, nfeatures_rna=0, nfeatures_adt=0,  hidden_rna=185,  hidden_adt=15, z_dim=20,classify_dim=17):
        """ Autoencoder for 2 modalities data (citeseq data and shareseq data) """
        super().__init__()
        self.encoder = Encoder(nfeatures_rna, nfeatures_adt, hidden_rna,  hidden_adt, z_dim)
        self.classify = nn.Sequential(nn.Linear(z_dim, classify_dim))
        self.decoder = Decoder(nfeatures_rna, nfeatures_adt, hidden_rna,  hidden_adt, z_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x_cty = self.classify(x)
        x = self.decoder(x)
        return x, x_cty
    
class DeepMerge_3modality(nn.Module):
    def __init__(self, nfeatures_rna=0, nfeatures_adt=0, nfeatures_atac=0,  hidden_rna=185,  hidden_adt=15, hidden_atac=185, z_dim=20,classify_dim=17):
        """ Autoencoder for 2 modalities data (citeseq data and shareseq data) """
        super().__init__()
        self.encoder = Encoder_3modality(nfeatures_rna, nfeatures_adt, nfeatures_atac, hidden_rna,  hidden_adt,hidden_atac, z_dim)
        self.classify = nn.Sequential(nn.Linear(z_dim, classify_dim))
        self.decoder = Decoder_3modality(nfeatures_rna, nfeatures_adt, nfeatures_atac, hidden_rna,  hidden_adt, hidden_atac, z_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x_cty = self.classify(x)
        x = self.decoder(x)
        return x, x_cty
    
