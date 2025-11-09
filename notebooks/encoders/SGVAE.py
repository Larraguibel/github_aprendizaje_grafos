import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def project_to_sphere(z, eps=1e-8):
    """Proyecta tensores a la esfera unidad fila a fila."""
    norm = torch.norm(z, dim=-1, keepdim=True) + eps
    return z / norm

def sample_hyperspherical(mu, noise_scale=0.1):
    """
    Muestreo aproximado tipo vMF:
    mu: direcciones unitarias (N, d).
    Se agrega ruido gaussiano y se reproyecta a la esfera.
    """
    if noise_scale <= 0.0:
        return project_to_sphere(mu)
    eps = torch.randn_like(mu) * noise_scale
    z = mu + eps
    return project_to_sphere(z)

def uniform_sphere_regularizer(z):
    """
    Regularizador que empuja la distribución hacia algo cercano a uniforme.
    Para distribución uniforme en la esfera, la media vectorial tiende a 0.
    Penalizamos la norma de la media.
    """
    mean_vec = z.mean(dim=0)
    return torch.sum(mean_vec**2)


class HypersphericalGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        mu = project_to_sphere(h)  # Direcciones unitarias
        return mu

class HypersphericalGraphVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64, noise_scale=0.1, lambda_uniform=1.0):
        super().__init__()
        self.encoder = HypersphericalGraphEncoder(in_dim, hidden_dim, latent_dim)
        self.noise_scale = noise_scale
        self.lambda_uniform = lambda_uniform

    def encode(self, x, edge_index):
        mu = self.encoder(x, edge_index)
        z = sample_hyperspherical(mu, self.noise_scale if self.training else 0.0)
        return mu, z

    def decode_logits(self, z, edge_index):
        # Producto interno para cada arista (similar a VGAE)
        src, dst = edge_index
        # (E,)
        logits = (z[src] * z[dst]).sum(dim=-1)
        return logits

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        """
        x: features nodales
        edge_index: grafo completo (para mensaje)
        pos_edge_index: aristas verdaderas muestreadas
        neg_edge_index: no-aristas muestreadas
        """
        mu, z = self.encode(x, edge_index)

        # logits para positivas y negativas
        pos_logits = self.decode_logits(z, pos_edge_index)
        neg_logits = self.decode_logits(z, neg_edge_index)

        # labels: 1 para positivas, 0 para negativas
        pos_labels = torch.ones(pos_logits.size(0), device=z.device)
        neg_labels = torch.zeros(neg_logits.size(0), device=z.device)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        recon_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Regularizador: empujar z hacia distribución uniforme en la esfera
        reg_uniform = uniform_sphere_regularizer(z)

        loss = recon_loss + self.lambda_uniform * reg_uniform

        return loss, recon_loss, reg_uniform, mu, z


CONFIGS = [
    # (nombre, params)
    (
        "A_lat32_h128_ns005_lu1e-3",
        dict(latent_dim=32, hidden_dim=128, noise_scale=0.05, lambda_uniform=1e-3)
    ),
    (
        "B_lat32_h128_ns010_lu1e-2",
        dict(latent_dim=32, hidden_dim=128, noise_scale=0.10, lambda_uniform=1e-2)
    ),
    (
        "C_lat32_h256_ns005_lu1e-2",
        dict(latent_dim=32, hidden_dim=256, noise_scale=0.05, lambda_uniform=1e-2)
    ),
    (
        "D_lat32_h256_ns010_lu1e-3",
        dict(latent_dim=32, hidden_dim=256, noise_scale=0.10, lambda_uniform=1e-3)
    ),
    (
        "E_lat64_h128_ns005_lu1e-2",
        dict(latent_dim=64, hidden_dim=128, noise_scale=0.05, lambda_uniform=1e-2)
    ),
    (
        "F_lat64_h128_ns010_lu1e-3",
        dict(latent_dim=64, hidden_dim=128, noise_scale=0.10, lambda_uniform=1e-3)
    ),
    (
        "G_lat64_h256_ns005_lu1e-3",
        dict(latent_dim=64, hidden_dim=256, noise_scale=0.05, lambda_uniform=1e-3)
    ),
    (
        "H_lat64_h256_ns010_lu1e-2",
        dict(latent_dim=64, hidden_dim=256, noise_scale=0.10, lambda_uniform=1e-2)
    ),
]