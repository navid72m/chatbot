const fs = require("fs");
const path = require("path");

exports.default = async function afterPack(context) {
  if (process.platform !== "darwin") return;

  const appOutDir = context.appOutDir;
  const appName = `${context.packager.appInfo.productFilename}.app`;

  const backendExe = path.join(
    appOutDir,
    appName,
    "Contents",
    "Resources",
    "backend",
    "chatbot-backend",
    "chatbot-backend"
  );

  try {
    fs.chmodSync(backendExe, 0o755);
    console.log("afterPack chmod 755:", backendExe);
  } catch (e) {
    console.warn("afterPack chmod failed:", backendExe, e);
  }
};
