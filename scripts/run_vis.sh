python3 scripts/vis_point_sim_3v3.py \
  --checkpoint runs/3v3_joint/3v3_joint_shot_1_gmh44ndf/best_model.pth \
  --data_dir data/point_sim_3v3/shot_npz \
  --num 10 --shuffle --seed 0 \
  --gap_frames 5 \
  --mode full \
  --obs_history 5 \
  --pass_thr 0.5 \
  --recv_control_mult 10 \
  --recv_control_sec 2.0 \
  --out output/vis_full_rand10_shot_1.gif

python3 scripts/vis_point_sim_3v3.py \
  --checkpoint runs/3v3_joint/3v3_joint_shot_3_qpwqbk1h/best_model.pth \
  --data_dir data/point_sim_3v3/shot_npz \
  --num 10 --shuffle --seed 0 \
  --gap_frames 5 \
  --mode full \
  --obs_history 5 \
  --pass_thr 0.5 \
  --recv_control_mult 10 \
  --recv_control_sec 2.0 \
  --out output/vis_full_rand10_shot_3.gif

python3 scripts/vis_point_sim_3v3.py \
  --checkpoint runs/3v3_joint/3v3_joint_shot_5_0bwchdey/best_model.pth \
  --data_dir data/point_sim_3v3/shot_npz \
  --num 10 --shuffle --seed 0 \
  --gap_frames 5 \
  --mode full \
  --obs_history 5 \
  --pass_thr 0.5 \
  --recv_control_mult 10 \
  --recv_control_sec 2.0 \
  --out output/vis_full_rand10_shot_5.gif