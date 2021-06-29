import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapred.lib.MultipleInputs;

public class exercise1 extends Configured implements Tool {
    public static class Map1 extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
        private Text keyword = new Text();

        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
            String line = value.toString();
            String[] words = line.split("\\s+");

            String word = words[0].trim();

            String yearStr = words[1].trim();
            int yearInt;
            boolean yearIsInt = true;
            try {  
                yearInt = Integer.parseInt(yearStr);  
            } catch(NumberFormatException e){  
                yearIsInt = false;
            } 

            String volume = words[3].trim();
            
            String[] targets = new String[] {"nu","chi","haw"}; 

            if (yearIsInt) {
                for (String target : targets){
                    if (word.toLowerCase().contains(target)){
                        keyword.set(yearStr+","+target);
                        output.collect(keyword, new Text(volume));
                    }
                }
            }            
        }
    }

    public static class Map2 extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
        private Text keyword = new Text();

        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
            String line = value.toString();
            String[] words = line.split("\\s+");

            String word1 = words[0].trim();
            String word2 = words[1].trim();

            String yearStr = words[2].trim();
            int yearInt;
            boolean yearIsInt = true;
            try {  
                yearInt = Integer.parseInt(yearStr);  
            } catch(NumberFormatException e){  
                yearIsInt = false;
            } 

            String volume = words[4].trim();
            
            String[] targets = new String[] {"nu","chi","haw"}; 
            
            if (yearIsInt) {
                for (String target : targets) {
                    boolean condition1 = word1.toLowerCase().contains(target);
                    boolean condition2 = word2.toLowerCase().contains(target);

                    if (condition1) {
                        keyword.set(yearStr+","+target);
                        output.collect(keyword,  new Text(volume));
                    }
                    if (condition2) {
                        keyword.set(yearStr+","+target);
                        output.collect(keyword,  new Text(volume));
                    }
                }
            }            
        }
    }

    public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
            Double sum = 0.0;
            int count = 0;

            while (values.hasNext()) {
                count++;
                sum += Double.parseDouble(values.next().toString());
            }

            Double mean = sum/count;

            output.collect(key, new Text(Double.toString(mean).trim()));
        }
    }
    
    public int run(String[] args) throws Exception {
        JobConf conf = new JobConf(getConf(), exercise1.class);
        conf.set("mapred.textoutputformat.separator", ",");
		conf.setJobName("hw1_exercise1_lf");

        conf.setMapOutputKeyClass(Text.class);
	    conf.setMapOutputValueClass(Text.class);
		conf.setOutputKeyClass(Text.class); 
		conf.setOutputValueClass(Text.class);

		conf.setMapperClass(Map1.class);
		conf.setMapperClass(Map2.class);
		conf.setReducerClass(Reduce.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

	    MultipleInputs.addInputPath(conf, new Path(args[0]), TextInputFormat.class, Map1.class);
	    MultipleInputs.addInputPath(conf, new Path(args[1]), TextInputFormat.class, Map2.class);
	    FileOutputFormat.setOutputPath(conf, new Path(args[2]));

		JobClient.runJob(conf);
		return 0;
    }
    
    public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new exercise1(), args);
		System.exit(res);
    }
}
