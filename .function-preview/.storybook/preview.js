import React from 'react'
import '@cognite/cogs.js/dist/cogs.css';
import 'antd/dist/antd.css';
import { ReactFlowProvider } from 'react-flow-renderer';

export const parameters = {
  actions: { argTypesRegex: "^on[A-Z].*" },
  controls: {
    matchers: {
      color: /(background|color)$/i,
      date: /Date$/,
    },
  },
}

export const decorators = [
  (Story) => (
    <div style={{ width: 'calc(100vw - 30px)', height: 'calc(100vh - 30px)' }}>
      <ReactFlowProvider>
        <Story />
      </ReactFlowProvider>
    </div>
  ),
];
