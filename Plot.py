def plot_plaws:
    xxs1 = [powerlaw(xx, plaw1[0][0], plaw1[0][1]) for xx in x1]
    xxs2 = [powerlaw(xx, plaw2[0][0], plaw2[0][1]) for xx in x2]

    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim([10**-3, 1])
    # plt.ylim([10**-4, 1])
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.plot(x1, xxs1,  linestyle='-', color='r')
    plt.plot(x2, xxs2,  linestyle='-', color='y')
    plt.show()

    print(str(plaw1))
    print(str(plaw2))

def plot():
    autocorr1 = pickle.load(open("autocorr1.p", "rb"))
    autocorr1 = autocorr1[:200]
    autocorr2 = pickle.load(open("autocorr2.p", "rb"))
    autocorr2 = autocorr2[:200]
    autocorr3 = pickle.load(open("autocorr3.p", "rb"))
    autocorr3 = autocorr3[:200]
    values = pickle.load(open("values.p", 'rb'))
    print(str(values[0]) + '\n')
    print(str(values[1]) + '\n')

    lag = np.arange(1, len(autocorr1) + 1)
    plt.stem(lag, autocorr1, basefmt='gray')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Returns')
    plt.show()
    plt.stem(lag, autocorr2, basefmt='gray')
    plt.title('Squared Returns')
    plt.show()
    plt.stem(lag, autocorr3, basefmt='gray')
    plt.title('Absolute Returns')
    plt.show()

def rando():
# fit = arch_model(norm_returns, vol="FIGARCH").fit()
# res = acorr_ljungbox(fit.std_resid**2, 100)
# res2 = fit.arch_lm_test(100, standardized=True)
# res3 = sc.jarque_bera(fit.std_resid)
# res4 = sc.shapiro(fit.std_resid)