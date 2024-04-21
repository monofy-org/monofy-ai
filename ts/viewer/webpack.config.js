const path = require("path");

module.exports = {
  mode: "production",
  entry: {
    viewer: "./src/viewer.ts",
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
    filename: "[name].js",
    path: path.resolve(__dirname, "public_html", "js"),
    library: "monofy",
    sourceMapFilename: "[name].js.map",    
  },
  devtool: "source-map",  
};
