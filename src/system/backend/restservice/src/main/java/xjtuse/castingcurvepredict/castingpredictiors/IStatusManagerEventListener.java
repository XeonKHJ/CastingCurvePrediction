package xjtuse.castingcurvepredict.castingpredictiors;

import xjtuse.castingcurvepredict.castingpredictiors.dummyimpl.StreamStatusManager;

public interface IStatusManagerEventListener {
    void onTaskStarting(StreamStatusManager statusManager);
    void onTaskStarted(StreamStatusManager statusManager);
    void onTaskStopped(StreamStatusManager statusManager);
}
