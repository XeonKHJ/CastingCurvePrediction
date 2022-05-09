package xjtuse.castingcurvepredict.castingpredictiors;

import xjtuse.castingcurvepredict.castingpredictiors.dummyimpl.StreamStatusManager;

public interface IStatusManagerEventListener {
    void onTaskStarting(IStatusManager statusManager);
    void onTaskStarted(IStatusManager statusManager);
    void onTaskStopped(IStatusManager statusManager);
}
