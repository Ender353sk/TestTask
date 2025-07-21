#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Coordinate {
    double lat;
    double lon;
    long long time;
};

void from_json(const json& j, Coordinate& c) {
    j.at("lat").get_to(c.lat);
    j.at("lon").get_to(c.lon);
    j.at("time").get_to(c.time);
}

void to_json(json& j, const Coordinate& c) {
    j = json{ {"lat", c.lat}, {"lon", c.lon}, {"time", c.time} };
}

double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    constexpr double R = 6371000.0; // Earth radius in meters
    double phi1 = lat1 * M_PI / 180.0;
    double phi2 = lat2 * M_PI / 180.0;
    double delta_phi = (lat2 - lat1) * M_PI / 180.0;
    double delta_lambda = (lon2 - lon1) * M_PI / 180.0;

    double a = sin(delta_phi/2) * sin(delta_phi/2) +
               cos(phi1) * cos(phi2) *
               sin(delta_lambda/2) * sin(delta_lambda/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));

    return R * c;
}

json process_coordinates(const json& input) {
    std::vector<Coordinate> coordinates = input.get<std::vector<Coordinate>>();
    json output;

    std::vector<Coordinate> corrected_points;
    int anomalies_detected = 0;
    int anomalies_corrected = 0;

    for (size_t i = 0; i < coordinates.size(); ++i) {
        if (i > 0 && i < coordinates.size() - 1) {
            double distance_prev = haversine_distance(coordinates[i].lat, coordinates[i].lon, coordinates[i - 1].lat, coordinates[i - 1].lon);
            double distance_next = haversine_distance(coordinates[i].lat, coordinates[i].lon, coordinates[i + 1].lat, coordinates[i + 1].lon);
            double speed_prev = distance_prev / (coordinates[i].time - coordinates[i - 1].time);
            double speed_next = distance_next / (coordinates[i + 1].time - coordinates[i].time);

            if (speed_prev > 200.0 || speed_next > 200.0) {
                anomalies_detected++;
                // Linear interpolation if previous and next points are valid
                Coordinate interpolated_point;
                interpolated_point.lat = (coordinates[i - 1].lat + coordinates[i + 1].lat) / 2;
                interpolated_point.lon = (coordinates[i - 1].lon + coordinates[i + 1].lon) / 2;
                interpolated_point.time = coordinates[i].time;
                corrected_points.push_back(interpolated_point);
                anomalies_corrected++;
            } else {
                corrected_points.push_back(coordinates[i]);
            }
        } else {
            corrected_points.push_back(coordinates[i]);
        }
    }

    output["corrected_points"] = corrected_points;
    output["anomalies_detected"] = anomalies_detected;
    output["anomalies_corrected"] = anomalies_corrected;

    return output;
}

int main() {
    // Example usage
    json input = R"([
        {"lat": 49588396, "lon": 34569212, "time": 1746025730},
        {"lat": 49588400, "lon": 34569220, "time": 1746025740},
        {"lat": 49588410, "lon": 34569230, "time": 1746025750}
    ])"_json;

    json output = process_coordinates(input);

    std::cout << output.dump(4) << std::endl;

    return 0;
}
