const path = require("path");

module.exports = {
  entry: "./src/viewer.ts",
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
    filename: "viewer.js",
    path: path.resolve(__dirname, "dist"),    
    library: "monofy",
    sourceMapFilename: "viewer.js.map",
  },
  devtool: "source-map",
};
