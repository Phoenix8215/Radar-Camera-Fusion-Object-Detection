#pragma once
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <map>
#include <math.h>

using namespace std;

class Radar_SF73 {
public:
    // Define the data format for 4D millimeter wave radar input
    struct feature
    {
        int id;
        float distance_x;
        float distance_y;
        float vlong;
        float vlat;
        float angle;
    };

    std::vector<feature> information_list;
    std::vector<feature> information_list_pre;
    // std::vector<feature> information_list_new;

    explicit Radar_SF73();

    ~Radar_SF73();

    void receive_data () noexcept;

private:
    // Static member variables: memory allocated during compilation stage
    // Declared inside class, initialized outside class
    // Static member variables are shared by all objects
    // static const achieves data sharing while maintaining immutability
    int Object_NofObjects;
    static constexpr float d_front2RearAxile = 5;
    static int s0; // CAN port socket descriptor
};
