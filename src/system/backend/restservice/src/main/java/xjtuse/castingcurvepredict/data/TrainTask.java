package xjtuse.castingcurvepredict.data;

import java.util.Calendar;

public class TrainTask {

    // TODO make all those members private;
    public int Id;
    public double Loss;
    public String Status;
    public String startTime;
    public String endTime;
    public int Epoch;
    public int ModelId;

    public int getId() {
        return Id;
    }

    public double getLoss() {
        return Loss;
    }

    public String getStatus() {
        return Status;
    }

    public String getStartTime() {
        return startTime;
    }

    public String getEndTime() {
        return endTime;
    }

    public int getEpoch() {
        return Epoch;
    }

    public int getModelId()
    {
        return ModelId;
    }
}
