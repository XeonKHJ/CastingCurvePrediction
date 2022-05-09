package xjtuse.castingcurvepredict.data;

import java.io.ObjectInputFilter.Status;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.utils.utils;

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

    public int getModelId() {
        return ModelId;
    }

    public TaskModel getInstance() {
        Date startTimeDate = null;
        Date endTimeDate = null;

        try {
            startTimeDate = utils.stringToDate(startTime);
            endTimeDate = utils.stringToDate(endTime);
        } catch (ParseException e) {

        } catch (NullPointerException e) {
            
        }

        TaskStatus status;
        if (Status.equals("Running")) {
            status = TaskStatus.Running;
        } else {
            status = TaskStatus.Created;
        }

        TaskModel model = new TaskModel(Id, Loss, Epoch, status, startTimeDate, endTimeDate);
        return model;
    }
}
