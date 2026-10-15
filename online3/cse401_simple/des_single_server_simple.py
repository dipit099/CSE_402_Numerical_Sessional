"""
CSE 401: Discrete-Event Simulation (DES) of a Single-Server Queue
Next-Event Time-Advance Algorithm (Simple Implementation)
"""

# ==============================================================================
# Single-Server Queue Simulation
# ==============================================================================
def simulate_single_server_queue(interarrival_times, service_times, num_delays_required):
    """
    Simulates a FIFO Single-Server Queue until num_delays_required customers
    complete their delay in queue.
    """
    ARRIVAL = 1
    DEPARTURE = 2
    INF = float("inf")
    
    # --- 1. Initialization Routine ---
    clock = 0.0
    server_status = 0     # 0 = Idle, 1 = Busy
    num_in_queue = 0
    time_arrival = []     # List of arrival times for waiting customers
    time_last_event = 0.0
    
    num_delayed = 0
    total_delay = 0.0
    area_num_in_q = 0.0
    area_server_status = 0.0
    
    delays = []
    trace_log = []
    
    # Event list
    arrival_idx = 0
    service_idx = 0
    
    # First arrival at clock + interarrival[0]
    time_next_arrival = clock + interarrival_times[arrival_idx]
    arrival_idx += 1
    time_next_departure = INF
    
    trace_log.append((clock, "Initialize", server_status, num_in_queue, num_delayed, area_num_in_q, area_server_status))
    
    # --- 2. Main Simulation Loop ---
    while num_delayed < num_delays_required:
        # Determine next event (Timing Routine)
        if time_next_arrival < time_next_departure:
            next_event_type = ARRIVAL
            clock = time_next_arrival
        else:
            next_event_type = DEPARTURE
            clock = time_next_departure
            
        # IMPORTANT: Accumulate areas BEFORE changing state variables!
        time_since_last_event = clock - time_last_event
        time_last_event = clock
        area_num_in_q += num_in_queue * time_since_last_event
        area_server_status += server_status * time_since_last_event
        
        # --- Event: ARRIVAL ---
        if next_event_type == ARRIVAL:
            # Schedule next arrival
            if arrival_idx < len(interarrival_times):
                time_next_arrival = clock + interarrival_times[arrival_idx]
                arrival_idx += 1
            else:
                time_next_arrival = INF
                
            # If server is idle, customer enters service immediately
            if server_status == 0:
                delay = 0.0
                delays.append(delay)
                total_delay += delay
                num_delayed += 1
                server_status = 1
                
                # Schedule departure
                time_next_departure = clock + service_times[service_idx]
                service_idx += 1
                event_desc = "Arrival (enters service, D=0)"
            else:
                # Server is busy, join waiting queue
                num_in_queue += 1
                time_arrival.append(clock)
                event_desc = "Arrival (joins queue)"
                
        # --- Event: DEPARTURE ---
        elif next_event_type == DEPARTURE:
            # If queue is empty, server becomes idle
            if num_in_queue == 0:
                server_status = 0
                time_next_departure = INF
                event_desc = "Departure (server idle)"
            else:
                # Remove first customer from queue (FIFO)
                num_in_queue -= 1
                arrival_time_of_cust = time_arrival.pop(0)
                delay = clock - arrival_time_of_cust
                delays.append(delay)
                total_delay += delay
                num_delayed += 1
                
                # Schedule next departure
                time_next_departure = clock + service_times[service_idx]
                service_idx += 1
                event_desc = f"Departure (next served, D={delay:.2f})"
                
        trace_log.append((clock, event_desc, server_status, num_in_queue, num_delayed, area_num_in_q, area_server_status))
        
    # --- 3. Compute Performance Measures ---
    avg_delay_in_queue = total_delay / num_delayed if num_delayed > 0 else 0.0
    time_avg_num_in_q = area_num_in_q / clock if clock > 0 else 0.0
    server_utilization = area_server_status / clock if clock > 0 else 0.0
    
    return {
        "delays": delays,
        "time_simulation_ended": clock,
        "average_delay_in_queue": avg_delay_in_queue,
        "time_average_number_in_queue": time_avg_num_in_q,
        "server_utilization": server_utilization,
        "area_num_in_q": area_num_in_q,
        "area_server_status": area_server_status,
        "trace": trace_log
    }


# ==============================================================================
# Main: Verification with Slide Example (6 Delays)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("Slide Example: Hand-Simulated Single-Server Queue (6 Delays)")
    print("=" * 75)
    
    interarrival = [0.4, 1.2, 0.5, 1.7, 0.2, 1.6, 0.2, 1.4, 1.9]
    service = [2.0, 0.7, 0.2, 1.1, 3.7, 0.6]
    
    res = simulate_single_server_queue(interarrival, service, num_delays_required=6)
    
    print(f"{'Clock':<7} | {'Event Description':<32} | {'Status':<6} | {'Queue':<5} | {'#Del':<4} | {'Area Q':<7} | {'Area B'}")
    print("-" * 80)
    for row in res["trace"]:
        print(f"{row[0]:<7.2f} | {row[1]:<32} | {row[2]:<6} | {row[3]:<5} | {row[4]:<4} | {row[5]:<7.2f} | {row[6]:.2f}")
        
    print("\n" + "=" * 45)
    print("SIMULATION SUMMARY REPORT")
    print("=" * 45)
    print(f"Delays D_i:                   {[round(d, 2) for d in res['delays']]}")
    print(f"Time Simulation Ended T(6):   {res['time_simulation_ended']:.2f}")
    print(f"Average Delay in Queue d(6):  {res['average_delay_in_queue']:.2f} (Slide: 0.95)")
    print(f"Time-Average Number in Q:     {res['time_average_number_in_queue']:.2f} (Slide: 1.15)")
    print(f"Server Utilization u(6):      {res['server_utilization']:.2f} (Slide: 0.90)")
