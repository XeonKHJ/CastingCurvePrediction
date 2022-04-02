package xjtuse.castingcurvepredict.data;

import java.util.Calendar;

public class TrainTask {
    int Id;
    double Loss;
    String Status;
    Calendar startTime;
    Calendar endTime;
    int epoch;

    public int getId() {
        return Id;
    }

    public double getLoss() {
        return Loss;
    }

    public String getStatus() {
        return Status;
    }

    public Calendar getStartTime() {
        return startTime;
    }

    public Calendar getEndTime() {
        return endTime;
    }

    public int getEpoch() {
        return epoch;
    }
}
