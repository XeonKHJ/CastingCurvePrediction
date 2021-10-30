using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using CsvHelper;

namespace CastingCurveDatasetCsvToJsonConverter
{
    class Program
    {
        private static string _datasetFolder = "../../../datasets/";
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            List<CastingCurveCsvItem> datasets = null;
            using (var reader = new StreamReader(_datasetFolder + "data.csv"))
            using (var csv = new CsvReader(reader, System.Globalization.CultureInfo.InvariantCulture))
            {
                var records = csv.GetRecords<CastingCurveCsvItem>();
                datasets = records.ToList();
            }
            List<CastingCurveJsonItem> jsonItems = new List<CastingCurveJsonItem>();
            CastingCurveJsonModel jsonModel = new CastingCurveJsonModel();
            for (int i = 0; i < datasets.Count; ++i)
            {
                jsonItems.Add(
                new CastingCurveJsonItem
                {
                    no = i + 1,
                    datetime = datasets[i].ds,
                    value = datasets[i].y
                });
            }
            jsonModel.CastingCurveValues = jsonItems.ToArray();
            var jsonString = JsonConvert.SerializeObject(jsonModel);

            File.WriteAllText(_datasetFolder + "data.json", jsonString);
        }
    }
}
