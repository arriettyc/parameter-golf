# Normal training (default run_id, no perf)
sh run_train.sh

# Custom run_id
sh run_train.sh my_experiment

# Custom run_id + perf metrics enabled
sh run_train.sh my_experiment 1




Nsight Systems capture (compiled, full speed, captures 5 steps):


# baseline
sh run_train.sh baseline

# your experiment with mfu
sh run_train.sh xin_exp 1 0 records/track_10min_16mb/2026-04-02_xin/train_gpt_xin.py


# → produces logs/my_run.nsys-rep
Then copy to your local machine and open in Nsight Systems GUI:


# on your local machine
scp xin@workstation:~/parameter-golf/logs/baseline_sp1024_v2.nsys-rep ./




### baseline
```bash
step:539/20000 val_loss:2.4545 val_bpb:1.4537 train_time:600618ms step_avg:1114.32ms
stopping_early: wallclock_cap train_time:600618ms step:539/20000
peak memory allocated: 10255 MiB reserved: 10574 MiB
Serialized model: 67224983 bytes
Code size: 55833 bytes
Total submission size: 67280816 bytes
Serialized model int8+zlib: 10295514 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 10351347 bytes
final_int8_zlib_roundtrip val_loss:2.4697 val_bpb:1.4627 eval_time:38427ms
final_int8_zlib_roundtrip_exact val_loss:2.46968863 val_bpb:1.46268872
```


### xin_exp
