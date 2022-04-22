package xjtuse.castingcurvepredict.castingpredictiors.impls;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import xjtuse.castingcurvepredict.castingpredictiors.GeneratorInput;
import xjtuse.castingcurvepredict.castingpredictiors.ICastingGenerator;
import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;

public class ConstCastingGenerator implements ICastingGenerator {

    @Override
    public CastingResultModel PredcitCastingCurve(GeneratorInput input) {
        // Read data from json
        CastingResultModel resultModel = new CastingResultModel();

        BufferedReader bfReader;
        try {
            bfReader = new BufferedReader(new FileReader("C:/Users/redal/source/repos/CastingCurvePrediction/datasets/data2.csv"));
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
                    double value = Double.parseDouble(item[1]);
                    resultModel.addResultItem(date, value);
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
    public void train() {
        // TODO Auto-generated method stub
        
    }

}
