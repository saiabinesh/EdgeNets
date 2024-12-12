@echo off
setlocal enabledelayedexpansion

set "remote_username=saiabinesh"
set "remote_host=kay.ichec.ie"
set "remote_folder=/ichec/work/nuig02/saiabinesh/EdgeNets/"

set "local_folder=results_classification_main\model_espnetv2_imagenet\aug_0.2_1.0\s_2.0_inp_224_sch_hybrid"

set "folders=20230507-011556"

for %%f in (%folders%) do (
    set "source_folder=!remote_folder!!local_folder!/%%f"
    set "destination_folder=!local_folder!\%%f"
    scp -r !remote_username!@!remote_host!:!source_folder! !destination_folder!
)
A#nd007.notverycommon now 