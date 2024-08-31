import sys
import pandas as pd


if __name__ == "__main__":

    # data_file = sys.argv[1]
    # data = pd.read_csv(data_file, sep=' ', skipinitialspace=True)

    data = pd.read_csv(sys.stdin, sep=' ', skipinitialspace=True)

    evids = []
    for i in data.index:
        subject = data["subject"][i]
        docid = data["docid"][i]
        time = int(data["time"][i])
        fROI = data["fROI"][i]

        evid = "{subject}_{docid}_{time}_{fROI}".format(subject=subject, docid=docid, time=time, fROI=fROI)
        evids.append(evid)


    data["evid"] = evids
    data.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')
