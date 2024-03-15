const path = require("path");

module.exports = {
  entry: "./src/api.ts",
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: {
          loader: "ts-loader",
          options: {
            compilerOptions: {
              sourceMap: true,
            },
          },
        },
        
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
  output: {
    filename: "api.js",
    path: path.resolve(__dirname, "dist"),    
    library: "monofy",
    sourceMapFilename: "api.js.map",
  },
  devtool: "source-map",
};
