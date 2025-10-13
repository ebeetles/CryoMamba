# 1. Background & Goals

Recent advances in Cryo-ET automation and direct electron detectors have dramatically increased the volume and resolution of tomograms, outpacing manual segmentation workflows. Traditional tools like UCSF ChimeraX excel at visualization but are not designed for scalable, automated segmentation.

The **CryoMamba** platform aims to deliver a **macOS desktop application** coupled with a **remote GPU inference server** to enable interactive visualization, automated segmentation using nnU-Net, and real-time feedback through multi-resolution previews. The solution is expected to **reduce segmentation turnaround time from hours to minutes**, **standardize results**, and **support both individual researchers and high-throughput core facilities**.

**Primary Goals:**
- Enable **interactive 3D visualization** of `.mrc` tomograms with real-time overlay of segmentation masks.  
- Provide a **simple interface** to configure and submit nnU-Net inference jobs to a remote GPU server.  
- Offer **multi-resolution previews** during inference for real-time feedback.  
- Support **manual editing** and **artifact filtering** to refine results.  
- Support **export** in standard formats (MRC, NIfTI, NRRD) and generate basic statistics.

**Target Users:**  
- Cryo-ET structural biologists  
- Computational biologists and image analysts  
- Core facility operators managing high-throughput Cryo-ET pipelines

---
