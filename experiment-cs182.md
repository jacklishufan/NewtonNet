# Readme #

To launch experiments, use 
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python newtonnet_train.py -c config_ani.yml -p ani
```

To try locally, run

```
CUDA_VISIBLE_DEVICES=0 python newtonnet_train.py -c config_ani.yml -p ani
```

after editing `config_ani.yml`  (`device: ` should be set to the actual number of gpus you have locally)