#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

//поскольку у нас уже все настроенно в zub, то я буду просто дополнять следующие файлы:

/*
./btc_no_vibrosi.csv
./data/csv/BTC_merged_2010_to_2025.csv
./data/csv/china_apply.csv
./data/json/spizhennoe_avg_size.json
./data/json/hash-rate-spizhennoe-v2.json
./data/json/active_count.json
./data/json/volume_sum.json
./data/json/transfers_volume_sum.json
./data/csv/zew.csv
./data/csv/gesi.csv
./data/csv/rubbles_dollars.csv
*/

std::string main_path = "../data/csv/main_data_copy.csv";

int main() {
    std::ifstream f(main_path);
    if(!f.is_open()){
        std::cerr << "error opening " << main_path << " file" << std::endl;
        return 404;
    }

    std::string line;
    while (std::getline(f, line)) {
        std::cout << line << std::endl;
    }
}