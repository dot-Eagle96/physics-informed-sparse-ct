# Sinogram-Based Tomography Densification and Denoising based on a Physics-Informed Deep Learning Approach

## How to run

For geometric dataset version:
- Run *generate_geometric_images.py* to generate the geometric images.
- Run *generate_geoemtric_dataset.py* to generate the sinograms of geometric dataset.
- Run *geometric_8x.py*.

For LoDoPaB-CT dataset versions:
- [Download](https://zenodo.org/records/3384092) and extract the LoDoPaB-CT dataset.
- Run *generate_lodopab_projections.py* to generate noise-free sinograms.
- Run the files you need that use this dataset (*lodopab_\*.py*).
