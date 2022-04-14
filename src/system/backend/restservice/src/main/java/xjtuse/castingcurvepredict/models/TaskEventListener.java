package xjtuse.castingcurvepredict.models;

import java.util.EventListener;

public interface TaskEventListener extends EventListener {
    void onTaskStarting(TaskModel task);
    void onTaskStarted(TaskModel task);
    void onTaskStopped(TaskModel task);
}
