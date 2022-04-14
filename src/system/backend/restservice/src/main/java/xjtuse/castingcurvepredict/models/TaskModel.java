package xjtuse.castingcurvepredict.models;

import java.util.Calendar;
import java.util.Vector;

public class TaskModel {

    private Vector<TaskEventListener> mListeners;
    private TaskStatus mStatus;
    private Calendar startTime;
    private Calendar stopTime;
    private int mId;

    public TaskModel(int id) {
        mId = id;
    }

    public void AddEventListener(TaskEventListener listener) {
        mListeners.add(listener);
    }

    public int getId() {
        return mId;
    }

    public void Start() {
        for (TaskEventListener listener : mListeners) {
            listener.onTaskStarting(this);
        }

        // TODO: start the task.

        for (TaskEventListener listener : mListeners) {
            listener.onTaskStarted(this);
        }
    }

    public void Stop() {
        // TODO: stop the task.

        for (TaskEventListener listener : mListeners) {
            listener.onTaskStopped(this);
        }
    }

    public void Pause() {

    }

    public void Update() {

    }
}