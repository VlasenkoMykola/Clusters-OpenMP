#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>

#include <omp.h>

//допоміжні функції
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        cols.push_back(item);
    }
    return cols;
}

//Зчитування CSV: автоматичний пошук стовпця Close
std::vector<double> read_prices(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string header;
    if (!std::getline(fin, header)) {
        throw std::runtime_error("Empty file or cannot read header: " + filename);
    }

    auto header_cols = split_csv_line(header);
    if (header_cols.empty()) {
        throw std::runtime_error("Header line is empty in: " + filename);
    }

    //Знаходимо індекс стовпця "Close" або "Adj Close"
    int close_idx = -1;
    for (int i = 0; i < (int)header_cols.size(); ++i) {
        std::string name = trim(header_cols[i]);
        for (auto &c : name) c = std::tolower(c);
        if (name == "close") {
            close_idx = i;
            break;
        }
    }
    if (close_idx == -1) {
        for (int i = 0; i < (int)header_cols.size(); ++i) {
            std::string name = trim(header_cols[i]);
            for (auto &c : name) c = std::tolower(c);
            if (name == "adj close" || name == "adjclose") {
                close_idx = i;
                break;
            }
        }
    }
    if (close_idx == -1) {
        //Резервний варіант: якщо формат "Date,Close", беремо другий стовпець (Close - ціна)
        if (header_cols.size() >= 2) {
            std::cerr << "Warning: 'Close' column not found by name, using column 1 (index 1)\n";
            close_idx = 1;
        } else {
            throw std::runtime_error("Cannot determine Close column in header: " + header);
        }
    }

    std::vector<double> prices;
    std::string line;
    int line_num = 1; //враховуємо рядок з заголовком

    while (std::getline(fin, line)) {
        ++line_num;
        if (line.empty()) continue;

        auto cols = split_csv_line(line);
        if ((int)cols.size() <= close_idx) {
            std::cerr << "Skipping line " << line_num
                      << " (not enough columns): " << line << "\n";
            continue;
        }

        std::string close_str = trim(cols[close_idx]);
        if (close_str.empty()) {
            std::cerr << "Skipping line " << line_num
                      << " (empty Close): " << line << "\n";
            continue;
        }

        try {
            double val = std::stod(close_str);
            prices.push_back(val);
        } catch (const std::exception& e) {
            std::cerr << "Skipping line " << line_num
                      << " (bad number '" << close_str
                      << "'): " << line << "\n";
            continue;
        }
    }

    if (prices.empty()) {
        throw std::runtime_error("No valid price data parsed from file: " + filename);
    }

    return prices;
}

//Прості рухомі середні (SMA - Simple Moving Average) з використанням OpenMP
std::vector<double> sma_omp(const std::vector<double>& data, int N) {
    int m = static_cast<int>(data.size());
    int out_len = m - N + 1;
    if (out_len <= 0) return {};

    std::vector<double> out(out_len);

    //Паралелимо зовнішній цикл: кожен потік обчислює свій елемент out[i]
    #pragma omp parallel for
    for (int i = 0; i < out_len; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += data[i + j];
        }
        out[i] = sum / N;
    }

    return out;
}

//Зважені рухомі середні (WMA - Weighted Moving Average) з використанням OpenMP
//Використовуємо ваги 1..N (більш пізні значення мають більшу вагу)
std::vector<double> wma_omp(const std::vector<double>& data, int N) {
    int m = static_cast<int>(data.size());
    int out_len = m - N + 1;
    if (out_len <= 0) return {};

    std::vector<double> out(out_len);
    double wsum = N * (N + 1) / 2.0;  //сума 1 + 2 + ... + N

    //Паралельне обчислення кожного вікна
    #pragma omp parallel for
    for (int i = 0; i < out_len; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            double w = j + 1;               //вага для data[i+j]
            sum += w * data[i + j];
        }
        out[i] = sum / wsum;
    }

    return out;
}

//Прогноз на один крок вперед та обчислення похибок
//
// data:              оригінальний ряд цін (x[0], x[1], ..., x[T-1])
// moving_averages:   вектор рухомих середніх (SMA або WMA)
// window_size:       довжина вікна, за яким рахували рухоме середнє
//
//  - вікно з індексом k охоплює елементи data[k] ... data[k + window_size - 1]
//  - кінець цього вікна відповідає моменту часу t = k + window_size - 1
//  - рухоме середнє moving_averages[k] — це оцінка "поточної" ціни на момент t
//  - ми використовуємо moving_averages[k] як прогноз ціни на наступний момент t+1,
//    тобто прогнозуємо data[t+1] = data[k + window_size]
//
// Отже:
//   y_pred (прогноз)  = moving_averages[k]
//   y_true (істина)   = data[k + window_size]
//   помилка           = y_true - y_pred
//
// Далі з усіх таких пар (y_true, y_pred) рахуємо MAE та MSE.
//
void evaluate_forecast(const std::vector<double>& data,
                       const std::vector<double>& moving_averages,
                       int window_size,
                       double& mae,
                       double& mse)
{
    const int data_len           = static_cast<int>(data.size());
    const int moving_averages_len = static_cast<int>(moving_averages.size());

    // Перевірка, що даних достатньо для будь-якого прогнозу
    if (moving_averages_len == 0 || data_len <= window_size) {
        mae = mse = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    double sum_abs_errors = 0.0;   // сума модулів похибок (для MAE)
    double sum_sq_errors  = 0.0;   // сума квадратів похибок (для MSE)
    int    count          = 0;     // кількість валідних пар (y_true, y_pred)

    // k — індекс вектора moving_averages[k]
    //
    // Вікно k охоплює:
    //   data[k] ... data[k + window_size - 1]
    //
    // Наступний момент часу:
    //   t_next = k + window_size
    //
    // Маємо гарантувати, що:
    //   t_next < data_len  (щоб існував y_true = data[t_next])
    //
    // Тому останній можливий k:
    //   k <= data_len - window_size - 1
    //
    // Також k не може виходити за межі масиву moving_averages:
    //   k <= moving_averages_len - 1
    //
    // Беремо мінімум цих двох обмежень.
    const int last_k_by_data   = data_len           - window_size - 1;
    const int last_k_by_ma     = moving_averages_len - 1;
    const int last_k           = std::min(last_k_by_data, last_k_by_ma);

    if (last_k < 0) {
        mae = mse = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    for (int k = 0; k <= last_k; ++k) {
        // Індекс наступного моменту часу, який ми прогнозуємо:
        // t_next = k + window_size
        const int t_next = k + window_size;

        // Фактичне значення ціни (y_true) — це data[t_next]
        const double y_true = data[t_next];

        // Прогноз (y_pred) — це рухоме середнє moving_averages[k]
        const double y_pred = moving_averages[k];

        const double error = y_true - y_pred;

        sum_abs_errors += std::abs(error);
        sum_sq_errors  += error * error;
        ++count;
    }

    if (count == 0) {
        mae = mse = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    mae = sum_abs_errors / count;
    mse = sum_sq_errors  / count;
}



//Бенчмарк: вимірювання часу для SMA та WMA при заданому N і числі потоків
//Тут ми ще й повертаємо MAE для SMA та WMA, щоб потім вибрати найкраще вікно.
struct BenchmarkResult {
    int threads;
    double sma_time;
    double wma_time;
    double sma_mae; // MAE для SMA при цьому N (той самий для будь-якої кількості потоків)
    double wma_mae; // MAE для WMA при цьому N
};

BenchmarkResult benchmark_one_N(const std::vector<double>& data,
                                int N,
                                int threads) {
    //Встановлюємо кількість потоків OpenMP
    omp_set_num_threads(threads);

    //Невеликий "розігрів" — перший виклик, щоб уникнути холодного кешу
    auto tmp_sma = sma_omp(data, N);
    (void)tmp_sma;

    BenchmarkResult res;
    res.threads = threads;

    //Вимірюємо час виконання обчислення SMA
    auto t1 = std::chrono::high_resolution_clock::now();
    auto sma = sma_omp(data, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    //Вимірюємо час виконання обчислення WMA
    auto wma = wma_omp(data, N);
    auto t3 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> d_sma = t2 - t1;
    std::chrono::duration<double> d_wma = t3 - t2;

    res.sma_time = d_sma.count();
    res.wma_time = d_wma.count();

    //Оцінюємо якість прогнозу (MAE і MSE) — цього разу без паралелізму, бо це швидка операція
    double mae_sma, mse_sma, mae_wma, mse_wma;
    evaluate_forecast(data, sma, N, mae_sma, mse_sma);
    evaluate_forecast(data, wma, N, mae_wma, mse_wma);

    res.sma_mae = mae_sma;
    res.wma_mae = mae_wma;

    std::cout << "  [threads = " << threads << "] "
              << "SMA_time (simple moving average runtime) = " << res.sma_time << " s, "
              << "WMA_time (weighted moving average runtime) = " << res.wma_time << " s, "
              << "SMA_MAE (simple moving average mean absolute error) = " << mae_sma << ", "
              << "WMA_MAE (weighted moving average mean absolute error) = " << mae_wma
              << std::endl;

    return res;
}

int main() {
    try {
        //Тут можна змінити файл на prices_weekly.csv або prices_monthly.csv
        std::string filename = "prices.csv";
        auto prices = read_prices(filename);
        std::cout << "Read " << prices.size()
                  << " price points from " << filename << std::endl;

        if (prices.size() < 200) {
            std::cerr << "Warning: not many data points, results may be unstable.\n";
        }

        std::cout << "\nLegend:\n"
                  << "  SMA = simple moving average\n"
                  << "  WMA = weighted moving average\n"
                  << "  MAE = mean absolute error of one-step-ahead forecast\n"
                  << "  threads = number of OpenMP threads used\n\n";

        //Набір розмірів вікна, для яких будемо проводити експерименти
        std::vector<int> window_sizes = {5, 10, 20, 50, 100};

        int max_threads = omp_get_max_threads();
        std::cout << "Max available threads (OpenMP) = " << max_threads << "\n";

        //Змінні для пошуку найкращого розміру вікна за точністю прогнозу (мінімальний MAE)
        double best_sma_mae = std::numeric_limits<double>::infinity();
        double best_wma_mae = std::numeric_limits<double>::infinity();
        int best_sma_N = -1;
        int best_wma_N = -1;

        for (int N : window_sizes) {
            if (N >= static_cast<int>(prices.size())) {
                std::cout << "\nWindow size N = " << N
                          << " is too large for data length, skipping.\n";
                continue;
            }

            std::cout << "\n=== Window size N = " << N
                      << " (number of points in moving average) ===\n";
            std::cout << "For each line below:\n"
                      << "  threads = OpenMP threads used\n"
                      << "  SMA_time = runtime for simple moving average\n"
                      << "  WMA_time = runtime for weighted moving average\n"
                      << "  SMA_MAE / WMA_MAE = forecast error for 1-step-ahead prediction\n\n";

            //Для поточного N запам'ятаємо MAE (беремо, наприклад, при threads = 1)
            double mae_sma_for_N = std::numeric_limits<double>::quiet_NaN();
            double mae_wma_for_N = std::numeric_limits<double>::quiet_NaN();

            for (int threads = 1; threads <= max_threads; ++threads) {
                auto res = benchmark_one_N(prices, N, threads);

                //MAE однаковий для будь-якої кількості потоків, тому достатньо зберегти один раз
                if (threads == 1) {
                    mae_sma_for_N = res.sma_mae;
                    mae_wma_for_N = res.wma_mae;
                }
            }

            //Оновлюємо загальний "найкращий" розмір вікна для SMA
            if (!std::isnan(mae_sma_for_N) && mae_sma_for_N < best_sma_mae) {
                best_sma_mae = mae_sma_for_N;
                best_sma_N = N;
            }

            //Оновлюємо загальний "найкращий" розмір вікна для WMA
            if (!std::isnan(mae_wma_for_N) && mae_wma_for_N < best_wma_mae) {
                best_wma_mae = mae_wma_for_N;
                best_wma_N = N;
            }
        }

        // Підсумковий висновок: найкращі N і найшвидші числа потоків
        std::cout << "\n=== Summary: best window sizes by forecast accuracy (minimum MAE) ===\n";

        if (best_sma_N != -1) {
            std::cout << "Best window for SMA (simple moving average): N = "
                      << best_sma_N
                      << ", MAE = " << best_sma_mae
                      << "\n  Fastest runtime for this N: "
                      << best_sma_time << " s at threads = " << best_sma_threads
                      << "\n";
        } else {
            std::cout << "Best window for SMA: not determined (no valid data).\n";
        }

        if (best_wma_N != -1) {
            std::cout << "Best window for WMA (weighted moving average): N = "
                      << best_wma_N
                      << ", MAE = " << best_wma_mae
                      << "\n  Fastest runtime for this N: "
                      << best_wma_time << " s at threads = " << best_wma_threads
                      << "\n";
        } else {
            std::cout << "Best window for WMA: not determined (no valid data).\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
