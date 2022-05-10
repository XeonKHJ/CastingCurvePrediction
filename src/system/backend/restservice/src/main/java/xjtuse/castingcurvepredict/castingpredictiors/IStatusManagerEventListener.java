package xjtuse.castingcurvepredict.castingpredictiors;

public interface IStatusManagerEventListener {
    void onTaskStarting(IStatusManager statusManager);
    void onTaskStarted(IStatusManager statusManager);
    void onTaskStopped(IStatusManager statusManager);
    void onTaskCompleted(IStatusManager statusManager);
}
