package xjtuse.castingcurvepredict.utils;

import java.util.Date;

import xjtuse.castingcurvepredict.castingpredictiors.LearningModelStatus;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;

import java.text.ParseException;
import java.text.SimpleDateFormat;

public class utils {

    public static String dateToString(Date date) {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        var str = simpleDateFormat.format(date);

        return str;
    }

    public static Date stringToDate(String dateString) throws ParseException {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        var date = simpleDateFormat.parse(dateString);

        return date;
    }

    public static TaskStatus StringToTaskStatus(String str) {
        switch (str) {
            case "Created":
                return TaskStatus.Created;
            case "Starting":
                return TaskStatus.Starting;
            case "Running":
                return TaskStatus.Running;
            case "Deleting":
                return TaskStatus.Deleting;
            case "Completed":
                return TaskStatus.Completed;
            default:
                return TaskStatus.Unknown;
        }
    }

    public static LearningModelStatus STringToModelStatus(String str) {
        switch (str) {
            case "Created":
                return LearningModelStatus.Created;
            case "Training":
                return LearningModelStatus.Training;
            case "Trained":
                return LearningModelStatus.Trained;
            default:
                return LearningModelStatus.Unknown;
        }
    }

}
