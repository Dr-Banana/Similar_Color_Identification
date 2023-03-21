import GUI
import tkinter as tk
import OtherMethod
import time
import DataGraph

path = "config/config1.jpg"
if __name__ == "__main__":
#     method = OtherMethod.ClusterMethod(path)
    # start_time = time.time()
    # k = method.GapStatistic()
    # print("gap statistic: ", '%.35f' % (time.time() - start_time), "K value:", k)
    # method.cluster_image(k, "gap statistic: ")
    #
    # start_time = time.time()
    # k = method.ElbowMethod()
    # print("Elbow Method: ", '%.35f' % (time.time() - start_time), "K value:", k)
    # method.cluster_image(k, "Elbow Method: ")
    #
    # start_time = time.time()
    # k = method.NoC()
    # print("NoC: ", '%.35f' % (time.time() - start_time), "K value:", k)
    # method.cluster_image(k, "NoC:")

    root = tk.Tk()
    gui = GUI.MyGUI(root)
    root.mainloop()
