from MCRT.visualize import PatchVisualizer
import os
print(1)
__root_dir__ = os.path.dirname(__file__)
model_path = "D:/Projects/MyProjects/MCRT/logs/finetune/charge_mobility/epoch=33-step=986.ckpt"
data_path = "D:/Projects/MyProjects/MCRT/MCRT/cifs/charge_mobility"

cifname = 'pentacene_CSP_1'
# cifname = 'T2_2_2225'# T2-beta
# cifname = 'T2_77_11225'# T2-alpha
# cifname = 'T2_2_3073'# T2-delta
# cifname = 'T2_173_7289'# T2-gamma
# make sure cif is in test split and its pickle is prepared

vis = PatchVisualizer.from_cifname(cifname, model_path, data_path,save_heatmap=True)
vis.draw_graph()
vis.draw_image_1d(top_n=10)
vis.draw_image_2d(top_n=10)