with open("train_top_80_uncorrected.txt", "r") as in_file:
    raw_lines = in_file.readlines() #sum(line.count("d") for line in in_file)
reversed_lines=[i for i in reversed(raw_lines[0:173])]
# print(reversed_lines)
start="Checkpoint saved at: "
end="\n"
for s in reversed(raw_lines):
    if start in s:
        print(s.split(start)[1])
        # exit()
        weights="./"+s.split(start)[1].split(end)[0]
        # end_time=time.time()
        break
print("path is : ",weights)
# exit()
# if os.path.exists(args.weights):
    # main(args)
# else:
    # print("path does not exist")
    # exit()
