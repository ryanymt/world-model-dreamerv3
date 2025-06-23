
# World Model : AI that dreams and learn from it

This is my attempt to run DreamerV3 of Danijar Hafner. Just my sketch notes, but something you can follow along if you want. I'll start with just simple preparation, and later on the detail configs and explanations (what i learnt)

You can read more about world models and how i come to this about on [my blog here](http://www.google.com). 

DreamerV3 paper can be [found here](https://arxiv.org/abs/2301.04104). 

Let's Start


## Pre-Requisites

Deploy a GKE autopilot cluster. I did in us-central1 just because it has almost every GPU / TPU. 

```bash
  gcloud container clusters create-auto <your_cluster_name> --region us-central1
```
And obtain your kubectl credentials

```bash
  gcloud container clusters get-credentials <cluster_name> --region us-central1 --project <project_id>
```

## Deployments

### 1. Let's prepare the image first 

1.1 First, let's git clone original dreamerV3 code 

```
 git clone https://github.com/danijar/dreamerv3
```

This code already provide the Dockerfile for you to build. But I created my own Dockerfile to have lesser things to install. But original one might give better performance. 

1.2. Create a Google Cloud Artifact registry(optional)

If you already have artifact registry and want to use it, feel free. 

```
 gcloud artifacts repositories create worldmodel \
    --repository-format=docker \
    --location=YOUR_REGION \
    --project=YOUR_PROJECT_ID
```
1.3. Build an image and push it to the registry
```
docker build -t YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/worldmodel/dreamerv3:latest .
docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/worldmodel/dreamerv3:latest
```
Well, now you have it. 

### 2. Deployment basic 
There are couple of config parameters I want to go through. You do not need to change the config.yaml, you can overwrite those from pod defination during the run time. . 

In config.yaml, main important sections are **default, run,** and **replay**. Other things like 
- **multicpu** is important if you are running on non-gpu machine and needs gpu emulation. 
- **jax** config parameters define which GPU you'll use for policy, which for training. 
- **environmental** parameters are for the different tasks. You can ignore for now
- **agent** parameters are very important depends on the task. 
  - main parameters we want to look at here is horizon and discount. These define how far ahead the model should look at. Longer horizon allows the agent to learn about the long-term consequences of its actions, which is crucial for tasks with delayed rewards. But it increase computational cost for training
  - the sub pramaters like encoder, decoder, dyn (rssm), reward head, cont head, more importantly policy (actor) and value (critic) are something i do not think i really have the complete understanding. So, you can dive the code and paper yourself 

- **default* parameters define basic configs like logs location and batch size
- **replay** parameters define the replay buffer
- **run** parameters are where to define how many steps to do in an environment, how frequent checkpoints should be saved, basically how the training and eval jobs will be run. 

Replay and run parameters will have pretty much impact on GPU and memory requirements. 

This is what i used in my runtime parameters
```
            python3.11 -m dreamerv3.main \
              --script parallel \
              --configs atari \
              --task atari_ms_pacman \
              --logdir /logs/pacman_l4_parallel \
              --run.save_every 5000 \
              --jax.train_devices 0 1 2 3 \
              --jax.policy_devices 0 \


              --run.steps 100000 \
              --run.log_every 10000 \
              --run.envs 32
```
Because i'm running on 4 x L4, i need parallelsim. It'll save a checkpoints at every 5000 steps, train on all 4 GPU but use only one for Policy (actor), and logs captured at every 10k steps. I can have run like 48 environments, but i tested with 32. 

Some of my eval run video recording can be found under the video. But here is the one of them. 
https://github.com/user-attachments/assets/51b12abd-cf07-4b8d-89bf-cf5bd96a2da2



