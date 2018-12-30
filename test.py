import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from scipy import signal
import seaborn as sns
from scipy import interpolate
from cycler import cycler


year = 2018
date0 = str(year - 1) + '-12-15'


def caltime(date1, date2):
    date1 = time.strptime(date1, "%Y-%m-%d")
    date2 = time.strptime(date2, "%Y-%m-%d")
    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    return (date2-date1).days


def get_mh_data(database_name):
    mh = {}
    db = sqlite3.connect(database_name).cursor()
    # print('Connect Successful!')
    db.execute("select * from sqlite_master")
    # print([x[1] for x in db.fetchall()])
    head = db.execute("pragma table_info([DATE_DATA])")
    for x in head.fetchall():
        mh[x[1]] = []

    table = db.execute("select * from DATE_DATA")
    for row in table:
        i = 0
        for k in mh:
            mh[k].append(row[i])
            i += 1
    return mh


def static_sleep(mh):
    dates = mh['DATE']
    day_nums = []
    slp_sts = []
    slp_eds = []

    for i in range(len(dates)):
        day_num = caltime(date0, dates[i])
        if 0 < day_num < 366:
            summary = mh['SUMMARY'][i]
            slp_st_loc = summary.find('"st":')
            slp_ed_loc = summary.find(',"ed":')
            slp_dp_loc = summary.find(',"dp":')
            slp_st = datetime.datetime.fromtimestamp(int(summary[slp_st_loc+5:slp_ed_loc]))
            slp_ed = datetime.datetime.fromtimestamp(int(summary[slp_ed_loc+6:slp_dp_loc]))
            if slp_ed != slp_st:
                day_nums.append(day_num)
                slp_sts.append((slp_st.hour + slp_st.minute/60 + slp_st.second/3600+8) % 24-8)
                slp_eds.append(slp_ed.hour + slp_ed.minute/60 + slp_st.second/3600)

    sns.jointplot(slp_sts, slp_eds, kind='reg')
    plt.plot(range(-4, 8), np.arange(-4, 8)+9, 'g')
    plt.plot(range(-4, 8), np.arange(-4, 8) + 7, 'r')
    plt.plot(range(-4, 8), np.arange(-4, 8) + 5, 'k')
    plt.text(6, 15, '9 hours', color='g')
    plt.text(6, 13, '7 hours', color='r')
    plt.text(6, 11, '5 hours', color='k')
    plt.xlabel('sleep_start')
    plt.ylabel('sleep_end')
    plt.axis([-8, 10, 0, 18])
    plt.show()


def static_step(mh):
    dates = mh['DATE']
    day_nums = []
    stp_ttls = []
    for i in range(len(dates)):
        date = dates[i]
        day_num = caltime(date0, date)
        if -50 < day_num < 366+50:
            summary = mh['SUMMARY'][i]
            stp_ttl_loc = summary.find('"ttl":')
            stp_dis_loc = summary.find(',"dis":')
            # print(summary[stp_ttl_loc+6:stp_dis_loc])
            stp_ttl = int(summary[stp_ttl_loc+6:stp_dis_loc])
            day_nums.append(day_num)
            stp_ttls.append(stp_ttl)
    step = np.array(stp_ttls)
    day = np.array(day_nums)
    return day, step


def step_cwt(day_nums, stp_ttls):
    f = interpolate.interp1d(day_nums, stp_ttls, kind='linear')
    start_day = day_nums[0]
    end_day = day_nums[-1]
    xnew = np.linspace(start_day, end_day, end_day-start_day+1)
    step = f(xnew)
    mean = stp_ttls.mean()
    print(mean)
    mean = step.mean()
    widths = np.arange(1, 40)
    cwtmatr = signal.cwt(step - mean, signal.ricker, widths)
    # sns.distplot(ttl, kde=False, bins=20)
    # plt.axis([0, 35000, 0, 0.0001])
    plt.imshow(cwtmatr,
               extent=[1, 365, 40, 5],
               cmap='jet', aspect=1,
               vmax=cwtmatr.max(), vmin=cwtmatr.min())

    date = [str(year)+'/'+str(x)+'/1' for x in range(1, 13)]
    plt.xticks(np.array([17, 48, 76, 107, 137, 168, 198, 229, 260, 290, 321, 351]), date)
    # plt.title("My Steps")
    # plt.xlabel("days from 2016/10/27")
    # plt.ylabel("Scale")
    # plt.colorbar()
    plt.show()
    # plt.savefig('figure\\step.png')


def static_7_day(days, steps):
    err = np.array([[0.]*7]*12)
    mean = np.array([[0.]*7]*12)
    # max = np.array([[0.] * 7] * 12)
    # min = np.array([[0.] * 7] * 12)
    alist = [[list()]*7]*12

    a1=list()
    for i in range(12):
        for j in range(7):
            for k in range(len(steps)):
                libai = (days[k] + 4) % 7
                date = datetime.datetime(2016, 12, 15) + datetime.timedelta(int(days[k]))
                # print(date)
                if int((date.month - 1)) == i and libai == j:
                    a1.append(steps[k])
                # alist[date.month - 1][libai].append(step[i])
            mean[i][j] = np.array(a1).mean()
            err[i][j] = np.array(a1).std()
            # max[i][j] = np.array(a1).max()
            # min[i][j] = np.array(a1).min()
            a1.clear()

    cmap = plt.cm.get_cmap('Paired')
    c = cycler('color', cmap(np.linspace(0, 1, 12)))
    plt.rcParams["axes.prop_cycle"] = c
    for i in range(12):
        ind = np.arange(7)
        # bottom = 50000*i
        # y = mean[i]+bottom
        # print(y)
        plt.bar(ind+0.3*(i % 3), mean[i], yerr=err[i], width=0.3, bottom=20000*int(i/3))
    plt.legend(range(1, 13), loc='center right')
    plt.xlim([-0.5, 8])
    plt.yticks(range(0, 90001, 10000), ['0', '10000', '0', '10000', '0', '10000', '0', '10000'])
    plt.xticks(np.arange(0, 8)+0.3, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()


def static_52_weeks(days, steps):
    first_date = time.strptime(date0, "%Y-%m-%d")
    anyday = int(datetime.datetime(first_date[0], first_date[1], first_date[2]).strftime("%w"))

    cld = [[0]*53 for x in range(7)]

    for i in range(len(days)):
        day = days[i]
        if 0 < day < 366:
            week_num = (day+anyday)//7
            day_of_week = (day+anyday) % 7
            cld[day_of_week][week_num] = steps[i]

    p = []
    for i in range(7):
        start = 0
        end = 53
        if cld[i][0] == 0:
            start = 1
        if cld[i][-1] == 0:
            end = 52
        y = np.array(cld[i][start:end])
        x = np.arange(start, end)
        N = 4
        # plt.scatter(range(start, end), y-i*20000)
        y_move = np.convolve(y, np.ones((N,)) / N)[(N - 1):]
        # plt.plot(range(start, end), y_move-i*20000)
        # plt.bar(x, y-y_move-i*20000)
        p.append(plt.bar(x, height=y-y_move, bottom=-i*20000+y*0))
        plt.plot(x, y*0-i*20000, c='k')
    plt.yticks(-np.arange(0, 7*20000, 20000), ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
    plt.xticks(np.arange(15, 365+15, 30)/7, range(1, 13))
    plt.xlim([-1, 53])
    plt.show()


if __name__ == '__main__':
    year = 2018
    file_name = 'apps/com.xiaomi.hm.health/db/origin_db_d05598c008a33f0ce464da4c71b90820'
    mh2018 = get_mh_data(file_name)
    # static_sleep(mh)
    day, step = static_step(mh2018)
    static_52_weeks(day, step)
