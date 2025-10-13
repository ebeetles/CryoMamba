# 🧰 Epic 4 — Visualization, Overlay & Export

## Summary
Upgrade desktop visualization to handle live previews, mask overlays, opacity controls, 3D rendering toggle, and export functions.

## Why
This is the user-facing core: real-time visualization during inference and exporting usable outputs.

## Scope
- Overlay layer management (preview → final swap)  
- Opacity slider, colormap selector, 3D volume toggle  
- Export mask to `.mrc`, NIfTI, PNG screenshots with scale bar

## Out of Scope
- Advanced manual editing

## Acceptance Criteria
- Previews appear live during inference; final mask replaces preview automatically  
- Export functions produce valid `.mrc` and `.png` files  
- Desktop remains ≥30 FPS on typical 512³ volumes

## Stories / Tasks
- Overlay controls & napari layer plumbing  
- WS listener → layer updater  
- Export menu & functions

## Dependencies
- Epics 1–3

## Milestone / Time Target
**Day 4**.
