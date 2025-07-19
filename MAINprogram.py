# coding: utf-8
import os
import json
import subprocess
import time
import openai
import statistics
import math
from typing import List, Dict, Tuple, Optional

# Конфігурація
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
CPP_COMPILER = "g++"
CPP_OUTPUT_EXECUTABLE_NAME = "gps_algorithm" # Ім'я вихідного виконуваного файлу

TEST_FILES = ["points.json", "points2.json", "points3.json"] # Список файлів з тестовімі даними

MAX_ITERATIONS = 15 # Максимальна кількість ітерацій
SUCCESS_ITERATIONS_TARGET = 15 # Цільова кількість успішних ітерацій

OUTPUT_DIR = "results" # Директорія для результатів

# Параметри валідації
ANOMALY_SPEED_THRESHOLD_M_PER_S = 200.0 # Поріг аномальної швидкості (м/с)
SPEED_DEVIATION_TOLERANCE_PERCENT = 35 # Допустиме відхилення швидкості (%)

class GPSDataProcessor:
    def __init__(self):
        self.iteration_results = []
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Ініціалізація клієнта OpenAI
        self.client = openai.OpenAI(api_key=OPENAI_KEY, timeout=30.0,)
        self.cpp_code_history = {} # Для збереження коду найкращих ітерацій

    def load_json_data(self, json_path: str) -> List[Dict]:
        """Завантажує JSON дані з файлу."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON файл повинен містити список об'єктів.")

            # Проста валідація структури точки
            for point in data:
                if not all(k in point for k in ["lat", "lon", "time"]):
                    raise ValueError(f"Некоректна структура точки в JSON: {point}")
            return data
        except FileNotFoundError:
            raise IOError(f"Файл не знайдено: {json_path}")
        except json.JSONDecodeError as e:
            raise IOError(f"Не вдалося декодувати JSON з {json_path}. Помилка: {e}")
        except ValueError as e:
            raise IOError(f"Некоректні дані в {json_path}. Помилка: {e}")
        except Exception as e:
            raise IOError(f"Не вдалося завантажити дані з {json_path}. Помилка: {e}")

    def generate_cpp_code(self, feedback: Optional[str] = None, iteration: int = 0) -> str:
        """Генерує С++ код за допомогою OpenAl."""
        base_prompt = f"""
Напиши алгоритм на С++ для виявлення та виправлення аномальних GPS координат.

Обов'язково включи функції `from_json` та `to_json` для структури `Coordinate` (визначеної як `struct Coordinate {{ double lat; double lon; long long time; }};`). Ці функції повинні бути визначені в глобальному просторі імен (або в тому ж просторі імен, що й `Coordinate`) для бібліотеки `nlohmann::json`, щоб забезпечити коректну десеріалізацію та серіалізацію `std::vector<Coordinate>` з JSON та в JSON. Ось приклад, який **обов'язково треба включити** у твій код:

// Функції для роботи з nlohmann/json та структурою Coordinate
void from_json(const nlohmann::json& j, Coordinate& c) {{
    j.at("lat").get_to(c.lat);
    j.at("lon").get_to(c.lon);
    j.at("time").get_to(c.time);
}}

void to_json(nlohmann::json& j, const Coordinate& c) {{
    j = nlohmann::json{{ {{"lat", c.lat}}, {{"lon", c.lon}}, {{"time", c.time}} }};
}}

Вимоги:
1. Вхід: JSON масив з координатами (lat, lon у форматі *1е6, тобто цілі числа, які
потрібно розділити на 10^6 для отримання десяткових градусів) та часом (timestamp,
unixtime у секундах).
2. Виявлення аномалій: точка вважається аномальною, якщо швидкість руху до/від цієї
точки перевищує {ANOMALY_SPEED_THRESHOLD_M_PER_S} М/С Швидкість повинна
розраховуватися між сусідніми точками.
3. Виправлення: лінійна інтерполяція аномальних точок на основі сусідніх коректних
точок.
Якщо аномальна точка знаходиться на початку або в кінці масиву, або якщо навколо
неї немає достатньої кількості коректних точок для інтерполяції, її слід залишити як є
або видалити (обери безпечніший підхід).
4. Вихід: JSON **об'єкт**, що обов'язково містить ключ "corrected_points" (масив виправлених точок)
та ключі зі статистикою, наприклад, "anomalies_detected" та "anomalies_corrected".
   **Приклад ОБОВ'ЯЗКОВОГО формату вихідного JSON:**
   {{
     "corrected_points": [
       {{"lat": 49588396, "lon": 34569212, "time": 1746025730}},
       ...
     ],
     "anomalies_detected": 2,
     "anomalies_corrected": 2
   }}


**ОБОВ'ЯЗКОВО:**
* Для використання `M_PI` (або інших математичних констант) **завжди включай `<cmath>` і перед ним `#define _USE_MATH_DEFINES`**, ось так:
    ```cpp
    #define _USE_MATH_DEFINES // Обов'язково перед #include <cmath>
    #include <cmath>
    ```
    Це забезпечить доступність `M_PI` на різних платформах.
* ** "Попередні версії коду часто мали помилки компіляції через відсутність #define _USE_MATH_DEFINES. Будь ласка, переконайся, що ти завжди його включаєш перед <cmath>."
* **НІКОЛИ не включай функцію `main()`** у генерований C++ код. Твій код має бути бібліотекою функцій, що приймає та повертає JSON рядки.
* **НІКОЛИ не вбудовуй приклади JSON даних** безпосередньо в C++ код (наприклад, використовуючи `R"([...)"`). Вхідні дані будуть надані через функцію.
* При ітерації по `std::vector` використовуй `size_t` або `std::vector<ТвійТип>::size_type` для лічильників циклів, щоб уникнути попереджень про порівняння знакових/беззнакових типів. Наприклад: `for (size_t i = 0; i < vec.size(); ++i)`.
* **Завжди огортай згенерований C++ код у блок markdown**, що починається з ````cpp` і закінчується ````. Жодного іншого тексту чи пояснень поза цим блоком бути не повинно.

Код повинен бути читабельним, ефективним та обробляти можливі помилки
вводу/виводу, некоректний JSON.
Використовуй стандартні бібліотеки С++.
Не використовуй сторонні бібліотеки для
парсингу JSON, напиши свій мінімальний парсер або використовуй `nlohmann/json`
(якщо це прийнятно для середовища, але краще без сторонніх залежностей для
простоти компіляції).
Якщо ти не можеш обійтися без сторонніх залежностей, вкажи це явно в коментарі на
початку коду та надай інструкцію з встановлення.
Використовуй формулу Гаверсина для розрахунку відстані між GPS координатами.
Надай лише готовий С++ код у блоці markdown, без зайвих пояснень поза блоком коду.
"""

        messages = [
            {"role": "system", "content": "You are a helpful C++ assistant. Generate only C++ code in a markdown block, and explain any non-standard libraries required for JSON parsing if used."},
            {"role": "user", "content": base_prompt}
        ]

        if feedback:
            messages.append({"role": "user", "content": f"Зворотний зв'язок:\n{feedback}\nБудь ласка, покращи код, приділяючи увагу продуктивності, коректності та обробці крайніх випадків (наприклад, аномалії, що йдуть поспіль, аномалії на початку/в кінці)."})

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
            )
            cpp_code = response.choices[0].message.content
            if "```cpp" in cpp_code:
                cpp_code = cpp_code.split("```cpp")[1].split("```")[0].strip()
            return cpp_code
        except openai.APIError as e:
            print(f"Помилка API OpenAI: {e}")
            return "ERROR_GENERATING_CODE"
        except Exception as e:
            print(f"Помилка при генерації коду: {e}")
            return "ERROR_GENERATING_CODE"

    def save_cpp_code(self, code: str, iteration: int) -> str:
        """Зберігає C++ код у файл."""
        filename = os.path.join(OUTPUT_DIR, f"gps_algorithm_{iteration}.cpp")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)
        self.cpp_code_history[iteration] = code # Зберігання коду для можливого фінального звіту
        return filename

    def compile_cpp_code(self, cpp_file: str) -> bool:
        """Компілює C++ код."""
        try:
            # Додаємо прапорці для оптимізації та попереджень "-std=c++17" для сучасних стандартів C++
            # "-O3" для максимальної оптимізації "-Wall -Wextra -pedantic" для включення всіх попереджень
            subprocess.run(
                [CPP_COMPILER, cpp_file, "-Iinclude", "-o", os.path.join(OUTPUT_DIR, CPP_OUTPUT_EXECUTABLE_NAME),
                 "-std=c++17", "-O3", "-Wall", "-Wextra", "-pedantic"],
                check=True,
                stderr=subprocess.PIPE,
                timeout=60 # Збільшення таймауту для компіляції на випадок великого коду
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Помилка компіляції:\n{e.stderr.decode('utf-8')}")
            return False
        except subprocess.TimeoutExpired:
            print("Перевищено таймаут компіляції.")
            return False
        except Exception as e:
            print(f"Невідома помилка при компіляції: {e}")
            return False

    def run_cpp_algorithm(self, input_file: str) -> Tuple[bool, float, Dict]:
        """Запускає скомпільований C++ код."""
        start_time = time.time()
        try:
            # Створення унікального тимчасового файлу для виводу, щоб уникнути конфліктів
            output_json_path = os.path.join(OUTPUT_DIR, f"output_{os.path.basename(input_file)}.json")

            # Передавання шляху до вхідного файлу як аргумент, чекаємо, що C++ код запише результат у файл або виведе в stdout.
            result = subprocess.run(
                [os.path.join(OUTPUT_DIR, CPP_OUTPUT_EXECUTABLE_NAME), input_file],
                capture_output=True,
                text=True,
                check=True,
                timeout=60 # Збільшення таймауту для виконання
            )
            execution_time = time.time() - start_time
            output_data = json.loads(result.stdout)
            return True, execution_time, output_data
        except json.JSONDecodeError:
            print(f"Некоректний вивід JSON з C++ програми:\n{result.stdout}")
            return False, 0.0, {"error": "Invalid JSON output from C++ program"}
        except subprocess.CalledProcessError as e:
            print(f"Помилка виконання C++ програми (код повернення {e.returncode}):\n{e.stderr}")
            return False, 0.0, {"error": f"Runtime error in C++ program: {e.stderr}"}
        except subprocess.TimeoutExpired:
            print("Перевищено таймаут виконання C++ програми.")
            return False, 0.0, {"error": "Timeout expired for C++ program"}
        except Exception as e:
            print(f"Невідома помилка при запуску C++ програми: {e}")
            return False, 0.0, {"error": f"Unknown error during C++ program execution: {e}"}

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Обчислює відстань між точками (формула Гаверсина)."""
        R = 6371000 # Радіус Землі в метрах

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def validate_results(self, original_data: List[Dict], processed_output: Dict) -> Tuple[bool, int, int, float, float, str]:
        """Перевіряє коректність виправлених даних."""
        corrected_data = processed_output.get("corrected_points", [])
        anomalies_detected = processed_output.get("anomalies_detected", 0)
        anomalies_corrected = processed_output.get("anomalies_corrected", 0)

        if not corrected_data:
            return False, anomalies_detected, anomalies_corrected, 0.0, 0.0, "Немає виправлених точок у виводі C++ програми."

        speeds = []

        # Перевірка швидкості у виправлених даних
        for i in range(1, len(corrected_data)):
            try:
                prev = corrected_data[i-1]
                curr = corrected_data[i]

                # Переконаємося, що lat/lon у виводі C++ також множить на 10^6, якщо AI їх не ділить
                lat1, lon1 = prev['lat'] / 1e6, prev['lon'] / 1e6
                lat2, lon2 = curr['lat'] / 1e6, curr['lon'] / 1e6

                time_diff = curr['time'] - prev['time']
                if time_diff <= 0:
                    continue

                distance = self.calculate_distance(lat1, lon1, lat2, lon2)
                speed = distance / time_diff
                speeds.append(speed)

                if speed > ANOMALY_SPEED_THRESHOLD_M_PER_S:
                    # Якщо після виправлення все ще є аномальні швидкості
                    return False, anomalies_detected, anomalies_corrected, 0.0, 0.0, \
                           f"Виявлено аномальну швидкість у виправлених даних: {speed:.2f} м/с між точками {i-1} та {i}."
            except KeyError as e:
                return False, anomalies_detected, anomalies_corrected, 0.0, 0.0, f"Відсутній ключ у даних виправлених точок: {e}"
            except Exception as e:
                return False, anomalies_detected, anomalies_corrected, 0.0, 0.0, f"Помилка при валідації виправлених даних: {e}"

        if not speeds:
            # Якщо після фільтрації немає даних для розрахунку швидкості, але точки є, це може бути ок.
            # Якщо points.json містить лише одну точку, speeds буде порожнім.
            if len(corrected_data) > 1:
                return False, anomalies_detected, anomalies_corrected, 0.0, 0.0, "Немає валідних інтервалів для розрахунку швидкості."
            else:
                return True, anomalies_detected, anomalies_corrected, 0.0, 0.0, "Менше двох точок для розрахунку швидкості."

        avg_speed = statistics.mean(speeds)
        median_speed = statistics.median(speeds)
        max_speed = max(speeds)

        # Перевірка відхилення максимальної швидкості від середньої
        # Умова "не сильно відрізняється від максимальної" може бути інтерпретована як "максимальна швидкість не сильно перевищує середню"
        if avg_speed > 0: # Уникнути ділення на нуль
            deviation = (max_speed - avg_speed) / avg_speed * 100
            if deviation > SPEED_DEVIATION_TOLERANCE_PERCENT:
                return False, anomalies_detected, anomalies_corrected, avg_speed, \
                       median_speed, f"Відхилення максимальної швидкості ({max_speed:.2f} м/с) від середньої ({avg_speed:.2f} м/с) занадто велике: {deviation:.2f}%."
        # Можна додати перевірку, що кількість точок не зменшилась занадто сильно,
        # якщо AI вирішує видаляти точки замість інтерполяції.
        # Наприклад: if len(corrected_data) < len(original_data) * 0.8:
        return True, anomalies_detected, anomalies_corrected, avg_speed, median_speed, ""

    def process_test_file(self, file_path: str, iteration: int) -> Dict:
        """Обробляє один тестовий файл."""
        results = {
            "test_file": file_path,
            "success": False,
            "exec_time": 0.0,
            "anomalies_detected": 0,
            "anomalies_corrected": 0,
            "avg_speed": 0.0,
            "median_speed": 0.0,
            "error": ""
        }
        try:
            original_data = self.load_json_data(file_path) # Використовуємо нову функцію

            # Зберігання вихідних даних у тимчасовий файл для передачі у C++ програму
            input_json_path = os.path.join(OUTPUT_DIR, f"input_{os.path.basename(file_path)}")
            with open(input_json_path, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, indent=2, ensure_ascii=False)

            success, exec_time, output = self.run_cpp_algorithm(input_json_path)
            results["exec_time"] = exec_time

            if not success:
                results["error"] = output.get("error", "Невідома помилка під час виконання C++ або виводу.")
                return results

            is_valid, anomalies_detected, anomalies_corrected, avg_speed, median_speed, error = \
                self.validate_results(original_data, output)

            results.update({
                "success": is_valid,
                "anomalies_detected": anomalies_detected,
                "anomalies_corrected": anomalies_corrected,
                "avg_speed": avg_speed,
                "median_speed": median_speed,
                "error": error
            })
            return results
        except IOError as e: # Для виявлення помилок завантаження файлів
            results["error"] = f"Помилка завантаження даних: {e}"
            return results
        except Exception as e:
            results["error"] = f"Невідома помилка обробки тестового файлу: {e}"
            return results

    def run_iteration(self, iteration: int, feedback: Optional[str] = None) -> Dict:
        """Виконує одну ітерацію."""
        iteration_summary = {
            "iteration": iteration,
            "compile_success": False,
            "test_results": [],
            "overall_success": False,
            "avg_exec_time": 0.0,
            "feedback": "",
            "cpp_code": "" # Додаємо для збереження коду в результати ітерації
        }

        print(f"Генерація C++ коду для ітерації {iteration}...")
        cpp_code = self.generate_cpp_code(feedback, iteration)

        if cpp_code == "ERROR_GENERATING_CODE":
            iteration_summary["feedback"] = "Помилка генерації коду AI. Перевірте ваш API ключ та доступ до моделі."
            return iteration_summary

        iteration_summary["cpp_code"] = cpp_code # Зберігання згенерованого коду
        cpp_file = self.save_cpp_code(cpp_code, iteration)

        print(f"Компіляція {cpp_file}...")
        compile_success = self.compile_cpp_code(cpp_file)
        iteration_summary["compile_success"] = compile_success

        if not compile_success:
            iteration_summary["feedback"] = "Помилка компіляції C++ коду."
            return iteration_summary

        total_exec_time = 0.0
        all_tests_passed = True
        print("Запуск тестів...")

        for test_file in TEST_FILES:
            test_result = self.process_test_file(test_file, iteration)
            iteration_summary["test_results"].append(test_result)
            total_exec_time += test_result["exec_time"]

            if not test_result["success"]:
                all_tests_passed = False
                print(f" Тест '{test_file}' не пройдено: {test_result['error']}")
            else:
                print(f" Тест '{test_file}' успішно пройдено. Час: {test_result['exec_time']:.4f} сек.")

        iteration_summary["overall_success"] = all_tests_passed
        iteration_summary["avg_exec_time"] = total_exec_time / len(TEST_FILES) if TEST_FILES else 0.0

        if all_tests_passed:
            iteration_summary["feedback"] = f"Всі тести успішно пройдено. Середній час виконання: {iteration_summary['avg_exec_time']:.4f} сек."
        else:
            errors = "\n".join([f"  {res['test_file']}: {res['error']}" for res in iteration_summary["test_results"] if not res["success"]])
            iteration_summary["feedback"] = f"Виявлено помилки в тестах:\n{errors}"

        print(f"Ітерація {iteration} завершена. Успіх: {all_tests_passed}. Середній час: {iteration_summary['avg_exec_time']:.4f} сек.")
        return iteration_summary

    def generate_final_report(self):
        """Генерує фінальний звіт."""
        report = {
            "total_iterations": len(self.iteration_results),
            "successful_iterations": sum(1 for it in self.iteration_results if it["overall_success"]),
            "best_iteration": None,
            "best_exec_time": float('inf'),
            "iterations": []
        }
        best_cpp_code = ""

        for it in self.iteration_results:
            report_entry = {
                "iteration": it["iteration"],
                "compile_success": it["compile_success"],
                "overall_success": it["overall_success"],
                "avg_exec_time": it["avg_exec_time"],
                "feedback": it["feedback"],
                "test_results_summary": [
                    {"test_file": tr["test_file"], "success": tr["success"], "exec_time": tr["exec_time"], "error": tr["error"]}
                    for tr in it["test_results"]
                ]
            }
            report["iterations"].append(report_entry)

            if it["overall_success"] and it["avg_exec_time"] < report["best_exec_time"]:
                report["best_iteration"] = it["iteration"]
                report["best_exec_time"] = it["avg_exec_time"]
                best_cpp_code = it["cpp_code"] # Зберігання коду найкращої ітерації

        report["best_algorithm_code"] = best_cpp_code # Додавання найкращого коду у звіт

        report_file = os.path.join(OUTPUT_DIR, "final_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n=== Фінальний звіт ===")
        print(f"Всього ітерацій: {report['total_iterations']}")
        print(f"Успішних ітерацій (всі тести пройдено): {report['successful_iterations']}/{report['total_iterations']}")

        if report["best_iteration"] is not None:
            print(f"Найкраща ітерація (за середнім часом виконання): Ітерація {report['best_iteration']} із середнім часом {report['best_exec_time']:.4f} сек.")
            print(f"\nКод найкращого алгоритму збережено у '{OUTPUT_DIR}/gps_algorithm_{report['best_iteration']}.cpp' та включено до звіту.")
        else:
            print("Не вдалося знайти ітерацій, що успішно пройшли всі тести.")

        print(f"Повний звіт збережено у: '{report_file}'")

        # Висновок про продуктивність
        if report["successful_iterations"] > 0 and report["total_iterations"] > 1:
            initial_successful_iterations = [it for it in self.iteration_results if it["overall_success"]][:1]
            if initial_successful_iterations:
                initial_avg_time = initial_successful_iterations[0]["avg_exec_time"]
                if report["best_exec_time"] < initial_avg_time:
                    print(f"\nВисновок: Вдалося досягти покращення продуктивності.")
                    print(f" Початковий середній час (перша успішна ітерація): {initial_avg_time:.4f} сек.")
                    print(f" Найкращий середній час: {report['best_exec_time']:.4f} сек.")
                    print(f" Покращення: {(initial_avg_time - report['best_exec_time']) / initial_avg_time * 100:.2f}%")
                elif report["best_exec_time"] > initial_avg_time:
                    print(f"\nВисновок: Продуктивність знизилася.")
                else:
                    print(f"\nВисновок: Продуктивність залишилася приблизно на тому ж рівні.")
            elif report["successful_iterations"] == 0:
                print("\nВисновок: Не вдалося отримати жодної повністю успішної ітерації, тому оцінити покращення продуктивності неможливо.")
            else:
                print("\nВисновок: Недостатньо успішних ітерацій для оцінки покращення продуктивності.")

    def run(self):
        """Запускає процес."""
        feedback = None
        successful_iterations_count = 0
        iteration_counter = 0 # Лічильник загальної кількості ітерацій
    
        # Виконуємо цикл, поки не наберемо потрібну кількість успішних ітерацій
        while successful_iterations_count < SUCCESS_ITERATIONS_TARGET:
            iteration_counter += 1
            print(f"\n--- Запуск ітерації {iteration_counter} (Ціль: {successful_iterations_count}/{SUCCESS_ITERATIONS_TARGET} успішних) ---")

            # Запобіжник, щоб уникнути нескінченного циклу і зайвих витрат
            if iteration_counter > 50: # Максимум 50 спроб
                print("Перевищено максимальну кількість загальних ітерацій (50). Зупинка.")
                break

            iteration_result = self.run_iteration(iteration_counter - 1, feedback)
            self.iteration_results.append(iteration_result)

            if iteration_result["overall_success"]:
                successful_iterations_count += 1
                print(f"Ітерація успішна! Всього успішних: {successful_iterations_count}")
        
            feedback = iteration_result["feedback"] # Передаємо зворотний зв'язок для наступної ітерації
            time.sleep(2) # Затримка, щоб уникнути лімітів API

        if successful_iterations_count >= SUCCESS_ITERATIONS_TARGET:
            print(f"\n🎉 Досягнуто цільової кількості ({SUCCESS_ITERATIONS_TARGET}) успішних ітерацій!")

        self.generate_final_report()

if __name__ == "__main__":
    if OPENAI_KEY == os.getenv("OPENAI_API_KEY"):
        processor = GPSDataProcessor()
        processor.run()
