"""
Practice Q6: Discrete-Event Simulation of a Barber Shop Queue (with Balking)
Simple, Student-Friendly Implementation
"""

import math

def simulate_barber_shop(interarrival_times, service_times, num_delays_required, capacity=None):
    """
    Simulates a single-server queue with optional waiting-room capacity (balking).
    """
    ARRIVAL, DEPARTURE, INF = 1, 2, float("inf")
    
    clock = 0.0
    server_status = 0   # 0 = Idle, 1 = Busy
    num_in_queue = 0
    time_arrival = []
    time_last_event = 0.0
    
    num_delayed = 0
    total_delay = 0.0
    area_q = 0.0
    area_b = 0.0
    max_queue = 0
    num_lost = 0
    
    delays = []
    trace = []
    
    # Scheduling first arrival
    arr_idx, serv_idx = 0, 0
    time_next_arrival = clock + interarrival_times[arr_idx]
    arr_idx += 1
    time_next_departure = INF
    
    trace.append((clock, "Initialize", server_status, num_in_queue, area_q, area_b))
    
    while num_delayed < num_delays_required:
        # Timing routine
        if time_next_arrival < time_next_departure:
            next_event = ARRIVAL
            clock = time_next_arrival
        else:
            next_event = DEPARTURE
            clock = time_next_departure
            
        # 1. Update Area Accumulators with OLD state variables!
        elapsed = clock - time_last_event
        time_last_event = clock
        area_q += num_in_queue * elapsed
        area_b += server_status * elapsed
        
        # 2. Process Events
        if next_event == ARRIVAL:
            # Schedule next arrival
            if arr_idx < len(interarrival_times):
                time_next_arrival = clock + interarrival_times[arr_idx]
                arr_idx += 1
            else:
                time_next_arrival = INF
                
            if server_status == 0:
                delays.append(0.0)
                num_delayed += 1
                server_status = 1
                time_next_departure = clock + service_times[serv_idx]
                serv_idx += 1
                desc = "Arrival (enters service, D=0)"
            elif capacity is not None and num_in_queue >= capacity:
                num_lost += 1
                desc = "Arrival (BALKS / Left)"
            else:
                num_in_queue += 1
                max_queue = max(max_queue, num_in_queue)
                time_arrival.append(clock)
                desc = "Arrival (joins queue)"
                
        elif next_event == DEPARTURE:
            if num_in_queue == 0:
                server_status = 0
                time_next_departure = INF
                desc = "Departure (server idle)"
            else:
                num_in_queue -= 1
                delay = clock - time_arrival.pop(0)
                delays.append(delay)
                total_delay += delay
                num_delayed += 1
                time_next_departure = clock + service_times[serv_idx]
                serv_idx += 1
                desc = f"Departure (served, D={delay:.2f})"
                
        trace.append((clock, desc, server_status, num_in_queue, area_q, area_b))
        
    return {
        "delays": delays,
        "T": clock,
        "avg_delay": total_delay / num_delayed if num_delayed else 0.0,
        "avg_queue": area_q / clock if clock else 0.0,
        "utilization": area_b / clock if clock else 0.0,
        "max_queue": max_queue,
        "num_lost": num_lost,
        "area_q": area_q,
        "area_b": area_b,
        "trace": trace
    }


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    interarrival = [0.8, 1.4, 0.3, 2.1, 0.9, 1.7, 0.4, 1.2, 2.6, 0.7]
    service = [2.1, 0.9, 1.8, 0.4, 2.7, 1.1, 0.6, 1.9, 0.8, 1.3]

    print("=" * 75)
    print("TASKS 1 & 2: Barber Shop Simulation Trace (8 Delays, Unlimited Capacity)")
    print("=" * 75)
    res = simulate_barber_shop(interarrival, service, num_delays_required=8)
    
    print(f"{'Clock':<7} | {'Event Description':<32} | {'Status':<6} | {'Queue':<5} | {'Area Q':<7} | {'Area B'}")
    print("-" * 75)
    for row in res["trace"]:
        print(f"{row[0]:<7.2f} | {row[1]:<32} | {row[2]:<6} | {row[3]:<5} | {row[4]:<7.2f} | {row[5]:.2f}")

    print("\n" + "=" * 45)
    print("RESULTS:")
    print("=" * 45)
    print(f"• Delays:                    {[round(d, 2) for d in res['delays']]}")
    print(f"• Time Ended T(8):           {res['T']:.2f} min")
    print(f"• (a) Average Delay d(8):    {res['avg_delay']:.4f} min")
    print(f"• (b) Time-Avg in Queue q(8):{res['avg_queue']:.4f}")
    print(f"• (c) Server Utilization:    {res['utilization']:.4f} ({res['utilization']*100:.1f}%)")
    print(f"• (d) Maximum Queue Length:  {res['max_queue']}")

    print("\n" + "=" * 75)
    print("TASK 3: Adding Waiting-Room Limit (Balking Capacity k = 2)")
    print("=" * 75)
    res_balk = simulate_barber_shop(interarrival, service, num_delays_required=8, capacity=2)
    print(f"With Capacity k=2 -> Avg Delay: {res_balk['avg_delay']:.4f}, Lost Customers: {res_balk['num_lost']}")

    print("\n" + "=" * 75)
    print("TASK 4: Key Viva & Reflection Points")
    print("=" * 75)
    print("1. Why areas must be updated BEFORE state variables:")
    print("   The time interval that just passed was spent at the OLD queue length.")
    print("   If state is changed first, the rectangle height becomes incorrect.")
    print("2. Why q(n) is a time-weighted integral and not a simple average of numbers seen:")
    print("   Queue length is continuous in time; a queue of 10 for 1 hour has different")
    print("   impact than a queue of 10 for 1 minute.")
