#include "radar/radar.h"
#include <linux/can.h>
#include <linux/can/raw.h>
#include <math.h>
#include <net/if.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <vector>

#define DEBUG 0
// Definition and memory allocation - no more memory allocation needed for future Radar_SF73 object creation
int Radar_SF73::s0;

Radar_SF73::Radar_SF73() {
    // Create socket for CAN communication
    s0 = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (s0 < 0) {
        perror("socket error");
        exit(-1);
    }

    // Bind socket to CAN device
    struct sockaddr_can addr = {0};
    struct ifreq ifr = {0};
    // Flexible CAN port setting - the following method is also acceptable
    sprintf(ifr.ifr_name, "can%d", 1);
    ioctl(s0, SIOCGIFINDEX, &ifr);
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    // Bind socket to can1 interface
    int ret = bind(s0, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0) {
        perror("bind error");
        close(s0);
        exit(-1);
    }

    // Set CAN frame filtering rules to accept specific message IDs
    struct can_filter rfilter[2];  // Define can_filter structure objects
                                   // Configure filter to only accept frames with specific IDs
    rfilter[0].can_id = 0x60A;
    rfilter[0].can_mask = CAN_SFF_MASK;
    rfilter[1].can_id = 0x60B;
    rfilter[1].can_mask = CAN_SFF_MASK;
    // Apply filter settings using setsockopt
    setsockopt(s0, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter));
}

Radar_SF73::~Radar_SF73() {
    close(s0);
}

/*
Collect four types of radar data:
- Target ID
- Target longitudinal distance  
- Target lateral distance
- Target longitudinal velocity
- Target lateral velocity
*/
void Radar_SF73::receive_data() noexcept {
    struct can_frame frame;
    while (true) {
        int nbytes = read(s0, &frame, sizeof(frame));  // Receive CAN frame
        if (nbytes < 0) {
            perror("can read error");
            exit(-1);
        }
        
        // Handle frame with ID 0x60A - contains object count information
        if (frame.can_id == 0x0000060A) {
            information_list_pre = information_list;
            information_list.clear();
            Object_NofObjects = frame.data[0];

        } 
        // Handle frame with ID 0x60B - contains object detail information
        else if ((frame.can_id == 0x0000060B) & (Object_NofObjects > 0)) {
            feature information;
            int Object_ID;
            Object_ID = frame.data[0];
            // Target ID
            information.id = Object_ID;
            
            // Union for efficient byte-to-value conversion
            union {
                unsigned short Object_DistLong;
                unsigned short Object_DistLat;  // y axis of radar1 is opposite
                unsigned short Object_VrelLong;
                unsigned short Object_VrelLat;
                unsigned char Object_DistLong_tmp[2];
                unsigned char Object_DistLat_tmp[2];
                unsigned char Object_VrelLong_tmp[2];
                unsigned char Object_VrelLat_tmp[2];
            } trans;

            // Parse longitudinal distance (Y-axis) from CAN frame bytes
            trans.Object_DistLong_tmp[1] = frame.data[1];
            trans.Object_DistLong_tmp[0] = frame.data[2];
            trans.Object_DistLong >>= 3;
            // Target longitudinal distance calculation with offset compensation
            information.distance_y = trans.Object_DistLong * 0.2 - 500;

            // Parse lateral distance (X-axis) from CAN frame bytes
            trans.Object_DistLat_tmp[1] = frame.data[2];
            trans.Object_DistLat_tmp[0] = frame.data[3];
            // Mask out upper bits (F occupies 4 bits)
            trans.Object_DistLat &= 0x07FF;  // 0000 0111 1111 1111
            // Target lateral distance calculation with offset compensation
            information.distance_x = trans.Object_DistLat * 0.2 - 204.6;

            // Parse longitudinal velocity from CAN frame bytes
            trans.Object_VrelLong_tmp[1] = frame.data[4];
            trans.Object_VrelLong_tmp[0] = frame.data[5];
            trans.Object_VrelLong &= 0xFFC0;  // 1111 1111 1100 0000
            trans.Object_VrelLong >>= 6;
            // Target longitudinal velocity calculation with offset compensation
            information.vlong = trans.Object_VrelLong * 0.25 - 128;

            // Parse lateral velocity from CAN frame bytes
            trans.Object_VrelLat_tmp[1] = frame.data[5];
            trans.Object_VrelLat_tmp[0] = frame.data[6];
            trans.Object_VrelLat &= 0x1FE0;  // 0011 1111 1110 0000
            trans.Object_VrelLat >>= 5;
            // Target lateral velocity calculation with offset compensation
            information.vlat = trans.Object_VrelLat * 0.25 - 64;

            // Extract dynamic property information
            unsigned char Object_DynProp;
            Object_DynProp = frame.data[6];
            Object_DynProp &= 0x07;

            // Radar cross section (RCS) value
            float Object_RCS = frame.data[7];

            // Adjust distance with front-to-rear axle offset
            information.distance_y = information.distance_y + d_front2RearAxile;

            information_list.emplace_back(information);
            
#if DEBUG
            cout << "information.id: " << information.id << endl;
            cout << "information.distance_x: " << information.distance_x << endl;
            cout << "information.distance_y: " << information.distance_y << endl;
            cout << "information.vlong: " << information.vlong << endl;
            cout << "information.vlat: " << information.vlat << endl;
            cout << "information.angle: " << information.angle << endl;
            /*
            Example output:
            information.id: 9
            information.distance_x: 4.6
            information.distance_y: 22
            information.vlong: 0
            information.vlat: 0
            information.angle: 0
            */
#endif
        } else {
            // Skip unrecognized frame IDs (equivalent to Python's pass)
            ;
        }
    }
}

// Static variable initialization outside class (recommended: initialize inside class)
// Note: const static and static const are equivalent, cannot directly initialize 
// non-integer constants inside class. Can modify int, bool, char, but not other types (double, float)
// In C++11, can use constexpr static or static constexpr to modify non-integer static member constants.
