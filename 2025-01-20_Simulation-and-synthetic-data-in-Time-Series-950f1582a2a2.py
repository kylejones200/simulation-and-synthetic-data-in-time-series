import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess


def plot_white_noise() -> None:
    np.random.seed(42)

    white_noise = np.random.normal(loc=0, scale=1, size=500)

    plt.figure(figsize=(10, 6))

    plt.plot(white_noise, label="Random Noise")

    plt.title("Simulated Random Noise")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("random_noise_simulation.png")

    plt.show()

    ar_params = [1, -0.7]

    ma_params = [1]

    ar_process = ArmaProcess(ar_params, ma_params)

    ar_data = ar_process.generate_sample(nsample=500)


def plot_ar_1_process() -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(ar_data, label="AR(1) Process")

    plt.title("Simulated AR(1) Process")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("ar1_simulation.png")

    plt.show()

    ar_params = [1]

    ma_params = [1, 0.5]

    ma_process = ArmaProcess(ar_params, ma_params)

    ma_data = ma_process.generate_sample(nsample=500)


def plot_ma_1_process() -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(ma_data, label="MA(1) Process")

    plt.title("Simulated MA(1) Process")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("ma1_simulation.png")

    plt.show()

    ar_params = [1, -0.5]

    ma_params = [1, 0.4]

    arima_process = ArmaProcess(ar_params, ma_params)

    arima_data = np.cumsum(arima_process.generate_sample(nsample=500))


def plot_arima_1_1_1_process() -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(arima_data, label="ARIMA(1,1,1) Process")

    plt.title("Simulated ARIMA(1,1,1) Process")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("arima_simulation.png")

    plt.show()

    time = np.arange(500)

    trend_series = 0.05 * time + np.random.normal(loc=0, scale=1, size=500)


def plot_the_data() -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(trend_series, label="Non-Stationary Series with Trend")

    plt.title("Simulated Time Series with Trend")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("trend_simulation.png")

    plt.show()

    "\n    Evaluating ARIMA on Simulated Data\n    "

    model = ARIMA(trend_series, order=(1, 1, 0))

    fit = model.fit()


def plot_the_original_series_and_model_predictions() -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(trend_series, label="Original Data")

    plt.plot(fit.fittedvalues, label="Fitted ARIMA Model", linestyle="--")

    plt.title("ARIMA Model on Simulated Data")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("arima_model_simulation.png")

    plt.show()


def main() -> None:
    plot_white_noise()
    plot_ar_1_process()
    plot_ma_1_process()
    plot_arima_1_1_1_process()
    plot_the_data()
    plot_the_original_series_and_model_predictions()


if __name__ == "__main__":
    main()
