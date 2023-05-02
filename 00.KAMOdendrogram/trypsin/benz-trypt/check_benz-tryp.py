fi = "filenames.lst"

lines = [x.strip() for x in open(fi, "r").readlines()]

def check_lig(line):
    benz = ["CPS4483-01", "CPS4483-02", "CPS4483-03", "CPS4483-04"]
    tryp = ["CPS4484-09", "CPS4484-10", "CPS4484-11", "CPS4484-12"]

    if any(puck_id in line for puck_id in benz):
        res = "benz"
    elif any(puck_id in line for puck_id in tryp):
        res  = "tryp"
    else:
        res = "unknown"

    return res

csv_out = "cc_clustering_res.csv"

csv_str = "idx,binding_compound,data\n"

for idx, line in enumerate(lines, 1):
    res = check_lig(line)
    print("Idx: %d, Compound: %s, Data: %s" % (idx, res, line))
    csv_str += "%d,%s,%s\n" % (idx, res, line)

with open(csv_out, "w") as fo:
    fo.write(csv_str)
