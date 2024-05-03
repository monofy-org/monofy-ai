const path = require("path");

module.exports = {
  mode: "production",
  entry: {    
    index: "./src/index.ts",
  },
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
    filename: "monofy-elements.js",
    path: path.resolve(__dirname, "dist", "js"),
    library: "monofy_elements",    
    sourceMapFilename: "monofy-elements.js.map",
  },
  devtool: "source-map",  
};
