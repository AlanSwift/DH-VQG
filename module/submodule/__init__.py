from module.submodule.gcn import GCNEncoder, GCNEncoderSpatial, GCNEncoderSpatialV1, GCNEncoderSpatialV2


str2gnn = {"gcn_spectral": GCNEncoder, "gcn_spatial_base": GCNEncoderSpatial,
           "gcn_spatial_v1": GCNEncoderSpatialV1, "gcn_spatial_v2": GCNEncoderSpatialV2}
__all__ = ["GCNEncoder", "GCNEncoderSpatial", "GCNEncoderSpatialV1", "GCNEncoderSpatialV2"]
