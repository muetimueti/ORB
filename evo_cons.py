import numpy as np
import evo.tools.file_interface as evof
from statistics import mean, median


def parser():
    import argparse
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("csv_tables", help="one or multiple csv result tables", nargs='+')
    main_parser.add_argument("--save", help="path to results file to save to", default=None)
    return main_parser


def load_from_csv(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1,
                         skip_footer=0, names=['mean', 'median', 'rmse'], usecols=(2, 3, 5))
    return data


if __name__ == "__main__":
    prs = parser()
    args = prs.parse_args()
    csv_data = [load_from_csv(f) for f in args.csv_tables]
    title = 'RPE'
    ref_name = 'groundtruth.txt'

    for i in range(len(csv_data)):
        est_name = args.csv_tables[i][-10:]
        res = evof.load_res_file('/home/ralph/SLAM/evo_dummy_result.zip')
        res.info["title"] = title
        res.info["est_name"] = est_name

        err_arr = []
        ts_arr = []

        for j in range(len(csv_data[i])):
            err_arr.append(csv_data[i][j][0])
            ts_arr.append(j)

        res.add_np_array("error_array", err_arr)
        res.add_np_array("timestamps", ts_arr)

        rmse_arr = []
        for j in range(len(csv_data[i])):
            rmse_arr.append(csv_data[i][j][2])

        statsdict = {"mean": mean(err_arr), "median": median(err_arr), "rmse": mean(rmse_arr)}

        res.add_stats(statsdict)

        save_path = args.save + est_name
        save_path = save_path[:-4] + ".zip"
        print("\nSaving consolidated results to:\n {}".format(save_path))

        evof.save_res_file(save_path, res)
