# Add Ps node
- The Ps node connect to Scheduler, registered in Scheduler, the Ps status: kJoining, the Cluster status: kJoining.
- Ps status become:kWorking & kProxying, get MetaData/old ps_router from Scheduler and update the ps_router. At same time the Scheduler notify the Worker the ps_router has been updated.
- Ps start a sep-thread to pull data from other Ps and proxy the worker's request.
- When Ps finish pull data, it will notify the Scheduler that Scheduler status become kNormal.

# Remove Ps node
- A ps try to leave from Cluster. The Cluster status become: kLeaving. Tell other Ps a node try to leave then other Ps status become: kWorking & kProxying. Notify the worker the ps_router has bee change.
- Other ps will try to pull data from the leaving Ps and proxy the request.
- When all Ps finish pull data. The Cluster will become to: kNormal and the leaving Ps can go safty.

# Register Model/Table
- A worker request to Scheduler the Scheduler will store the MetaData, assign a unique Model/Table id. And tell the Ps to Register a Model/Table.

# Save Checkpoint
- A worker request to Scheduler to save a checkpoint. The Scheduler will send MetaData/ps_router to All Ps and The Ps will dump the checkpoint.