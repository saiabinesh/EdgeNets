import time
start_time=time.time()
with open("/ichec/work/ngcom027c/logs/train_imagenet_classification_top70_and_80_90_100.sh", "r") as in_file:
    raw_lines = in_file.readlines() #sum(line.count("d") for line in in_file)
start="Checkpoint saved at: "
end="\n"
for s in raw_lines:
    if start in s:
        print((s.split(start))[1].split(end)[0])
        end_time=time.time()
        break
print("total_time = ",(end_time-start_time))
