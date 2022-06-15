package xjtuse.castingcurvepredict.castingpredictiors.oldimpls;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import xjtuse.castingcurvepredict.castingpredictiors.GeneratorInput;
import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;
import xjtuse.castingcurvepredict.models.TaskModel;

public class ConstCastingGenerator implements ICastingGenerator {

    @Override
    public CastingResultModel PredcitCastingCurve(GeneratorInput input) {
        // Read data from json
        CastingResultModel resultModel = new CastingResultModel();

        BufferedReader bfReader;
        try {
            bfReader = new BufferedReader(new FileReader("C:/Users/redal/source/repos/CastingCurvePrediction/datasets/constdata.csv"));
            Boolean firstLine = true;
            String line;
            while((line = bfReader.readLine()) != null)
            {
                if(firstLine)
                {
                    firstLine = false;
                }
                else{
                    String itemToSplit = line;
                    String[] item = itemToSplit.split(",");
                    String date = item[0];
                    double lv_act = Double.parseDouble(item[2]);
                    double std_pos = Double.parseDouble(item[3]);
                    double tudishWeight = Double.parseDouble(item[5]);
                    double ladleWeight = Double.parseDouble(item[6]);
                    resultModel.addResultItem(date, std_pos, lv_act, tudishWeight, ladleWeight);
                }
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return resultModel;
    }

    @Override
    public void updateModel(CastingModel data) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void updateModel(ArrayList<CastingModel> datas) {
        // TODO Auto-generated method stub
        
    }


    @Override
    public void train(TaskModel task) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void stop(TaskModel task) {
        // TODO Auto-generated method stub
        
    }

}
