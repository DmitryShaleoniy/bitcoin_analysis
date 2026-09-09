#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <curl/curl.h>

//поскольку у нас уже все настроенно в zub, то я буду просто дополнять следующие файлы:

/*
csv:
./btc_no_vibrosi.csv
./data/csv/BTC_merged_2010_to_2025.csv
./data/csv/china_apply.csv
./data/csv/zew.csv
./data/csv/gesi.csv
./data/csv/rubbles_dollars.csv

json:
./data/json/spizhennoe_avg_size.json
./data/json/hash-rate-spizhennoe-v2.json
./data/json/active_count.json
./data/json/volume_sum.json
./data/json/transfers_volume_sum.json
*/

/*
errors:
    404 - main_data file not found
    4041 - error opening close_open_value file
    420 - error initializing curl object
*/

std::string main_path = "../data/csv/main_data_copy.csv";

//прямые ссылки для скачивания

std::string close_open_value_download_url = "https://img.bgstatic.com/multiLang/coinHistory/bitcoin.xlsx"; //она скачаивает просто xlsx файл с фиксированным временным промежутком - один год каждый день

//функции для обработки данных

size_t close_open_value_handaling(void* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total_size = size * nmemb;
    //сейчас будем писать в файл:

    std::ofstream* stream = static_cast<std::ofstream*>(userdata);

    stream->write(static_cast<char*>(ptr), total_size);

    if(!stream->good()){
        std::cerr << "error writing to close_open_value_file";
        return 0;
    }

    return total_size; // количество обработанных байт
    // если вернуть меньше чем получили - будет ошибка CURL_WRITE_ERROR
}

int main() {
    std::ifstream f(main_path);
    if(!f.is_open()){
        std::cerr << "error opening " << main_path << " file" << std::endl;
        return 404;
    }
    
    //тут просто вывод главного файла

    // std::string line;
    // while (std::getline(f, line)) {
    //     std::stringstream ss(line);
    //     std::string cell;
    //     //std::getline(ss, cell, ' ')
    //     while(ss >> cell){
    //         std::cout<< cell<< " " << std::endl;
    //     }
    // }

    f.close();

    //начнем с csv файлов
    //btc_no_vibrosi.csv имеет атрибуты: open	close	volume	date    
    CURL* curl = curl_easy_init(); //будем курлить ссылками на скачивание, где есть эта  нативная ссылка на скачивание

    if(!curl) {
        std::cerr << "error initializing curl object" << std::endl;
        return 420;
    }

    std::ofstream c_o_v_fp("./downloaded_data/close_open_value_downloaded.csv");
    if(!c_o_v_fp.is_open()){
        std::cerr << "error opening close_open_value file" << std::endl;
        return 4041;
    }

    //натсройки подключения
    //curl_easy_setopt(curl, CURLOPT_URL, close_open_value_download_url.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, "https://www.coingecko.com/price_charts/export/bitcoin/usd.csv");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, close_open_value_handaling);
    //curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, "https://www.coingecko.com/price_charts/export/bitcoin/usd.csv");

    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &c_o_v_fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");

    auto res = curl_easy_perform(curl);

    long http_code = 0;

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    
    std::cout << "Скачивание завершено. Код: " << http_code 
              << ", Ошибка CURL: " << curl_easy_strerror(res) << std::endl;
    
    curl_easy_cleanup(curl);

    c_o_v_fp.close();


}