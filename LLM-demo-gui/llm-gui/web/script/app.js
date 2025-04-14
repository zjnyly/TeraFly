const { ref } = Vue;
const { createApp } = Vue;
const { useQuasar } = Quasar;

var $q;

const avatarList = {
  sysu: "img/sysu.jpeg",
  gpu: "img/gpu.png",
  fpga: "img/fpga.png",
  cpu: "img/cpu.png",
};

const dialogContent = ref([]);

// 连接相关

const connectState = ref("disconnected");
const messageSending = ref(false);
const displayConnectPanel = ref(false);
const connectAddressInput = ref("localhost:10088");
const showConnectFailedDialog = ref(false);
var websocketConnection = null;

function connectServer(resolve, reject) {
  websocketConnection = new WebSocket(`ws://${connectAddressInput.value}`);

  websocketConnection.onopen = () => {
    connectState.value = "connected";
    resolve();

    $q.notify({
      color: "positive",
      message: "连接成功",
      icon: "check",
      position: "top",
    });
  };

  websocketConnection.onmessage = (event) => {
    websocketOnMessage(JSON.parse(event.data));
  };

  websocketConnection.onclose = () => {
    setErrorMessageBubble("连接已关闭");

    if (connectState.value === "connected") {
      $q.notify({
        message: "连接已关闭",
        icon: "info",
        position: "top",
      });
    }

    connectState.value = "disconnected";
  };

  websocketConnection.onerror = (error) => {
    reject(error);
    connectState.value = "disconnected";
    setErrorMessageBubble("连接失败");
  };
}

function handleConnectButtonClick() {
  connectState.value = "connecting";

  const promise = new Promise((resolve, reject) => {
    connectServer(resolve, reject);
  });

  promise
    .then(() => {
      connectState.value = "connected";
      showConnectFailedDialog.value = false;
      messageSending.value = false;
    })
    .catch(() => {
      connectState.value = "disconnected";
      showConnectFailedDialog.value = true;
    });
}

function handleDisconnectButtonClick() {
  websocketConnection.close();
}

// 消息发送

const messageInput = ref("");

function handleMessageSendButtonClick() {
  const content = {
    type: "input",
    content: messageInput.value,
  };

  const display = {
    sent: true,
    content: messageInput.value,
    name: "",
    stamp: "",
    avatar: avatarList["sysu"],
    completed: true,
  };

  dialogContent.value.push(display);
  websocketConnection.send(JSON.stringify(content));

  messageSending.value = true;
  messageInput.value = "";
}

function onResponseStart(eventData) {
  dialogContent.value.push({
    sent: false,
    content: "",
    name: eventData.displayName,
    stamp: "接收中...",
    avatar: avatarList[eventData.avatarType],
    completed: false,
  });

  const ack = {
    type: "acknowledge",
    originalType: "responseStart",
  };

  websocketConnection.send(JSON.stringify(ack));
}

function onResponse(eventData) {
  const length = dialogContent.value.length;
  dialogContent.value[length - 1].content += eventData.content;

  const ack = {
    type: "acknowledge",
    originalType: "response",
  };

  websocketConnection.send(JSON.stringify(ack));
}

function onResponseEnd(eventData) {
  const length = dialogContent.value.length;

  dialogContent.value[length - 1].stamp = `平均延迟: ${Math.round(eventData.latency)}ms`;
  dialogContent.value[length - 1].completed = true;

  const ack = {
    type: "acknowledge",
    originalType: "responseEnd",
  };

  websocketConnection.send(JSON.stringify(ack));

  messageSending.value = false;
}

function setErrorMessageBubble(text) {
  const length = dialogContent.value.length;

  if (length === 0) return;
  if (dialogContent.value[length - 1].completed) return;

  dialogContent.value[length - 1].content = text;
  dialogContent.value[length - 1].completed = true;
  dialogContent.value[length - 1].stamp = "";
}

function websocketOnMessage(eventData) {
  const messageType = eventData.type;

  switch (messageType) {
    case "responseStart":
      onResponseStart(eventData);
      break;

    case "response":
      onResponse(eventData);
      break;

    case "responseEnd":
      onResponseEnd(eventData);
      break;
  }
}

// 辅助

function copyToClipboard(text) {
  navigator.clipboard.writeText(text);
}

// 启动Vue和Quasar框架

function variableSetup() {
  $q = useQuasar();

  return {
    connectState,
    dialogContent,
    displayConnectPanel,
    connectAddressInput,
    messageSending,
    showConnectFailedDialog,
    messageInput,
    handleConnectButtonClick,
    handleDisconnectButtonClick,
    handleMessageSendButtonClick,
  };
}

const quasarConfig = {
  dark: true,
};

const app = createApp({
  setup: variableSetup,
});

app.use(Quasar, { config: quasarConfig });
app.mount("#q-app");
