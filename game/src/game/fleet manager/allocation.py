#read incomming messages from redis
request allocation hash, allocationresults stream

if message type is allocation request
    roi=extract(message)
    result=allocation_process(roi)
    send result to redis
allocation_process(roi):
    # process the roi and return the result
redis.xadd(
    "allocation_results",
    {
        "agent_id": player.id,
        "time": current_time,
        "status": 0,
        "roi": "",
    }
)
